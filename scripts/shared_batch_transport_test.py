"""CPU checks for byte-preserving shared-memory batch transport."""
import gc
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.data import dataloaders as dl
from openpi.data import transforms
from streaming_accumulation import StreamingAccumulator
from streaming_accumulation_test import fixture

torch.set_num_threads(1)


def numpy_collate(items):
    return jax.tree.map(lambda *xs:np.stack([np.asarray(x) for x in xs]),*items)


class Samples:
    def __len__(self):return 32
    def __getitem__(self,i):
        readonly=np.arange(6,dtype=np.float32);readonly.flags.writeable=False
        return {'id':np.int64(i),'features':np.arange(4,dtype=np.float32)/7+i/11,
                'labels':np.arange(4,dtype=np.float32)/3-i/17,
                'image':(np.arange(72,dtype=np.uint8).reshape(2,3,4,3)+i),
                'mask':np.bool_(i%2),'tokens':np.arange(5,dtype=np.int32)+i,
                'float16':np.arange(3,dtype=np.float16),
                'scalar':3.25,'boolean':True,'integer':4,
                'views':(readonly,np.arange(6,dtype=np.int16)[::-1],np.arange(12,dtype=np.float64).reshape(3,4).T),
                'optional':None}


def cleanup(loader):
    iterator=getattr(loader,'_iterator',None)
    if iterator is not None:iterator._shutdown_workers()


def assert_bytes(test,a,b):
    test.assertEqual(jax.tree.structure(a),jax.tree.structure(b))
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):
        x=np.asarray(x);y=dl._as_numpy(y)
        test.assertEqual(x.shape,y.shape);test.assertEqual(x.dtype,y.dtype)
        test.assertEqual(x.tobytes(),y.tobytes())


class SharedBatchTests(unittest.TestCase):
    def test_leaf_shapes_types_and_bytes(self):
        items=[Samples()[i] for i in range(8)]
        a,b=numpy_collate(items),dl._collate_fn(items)
        self.assertTrue(all(isinstance(x,torch.Tensor) for x in jax.tree.leaves(b)))
        assert_bytes(self,a,b)
        for item in items:self.assertFalse(item['views'][0].flags.writeable)

    def test_numpy_view_has_no_extra_array_copy(self):
        x=torch.arange(16,dtype=torch.float32).share_memory_()
        a=dl._as_numpy(x)
        self.assertTrue(np.shares_memory(a,x.numpy()))
        self.assertEqual(a.__array_interface__['data'][0],x.data_ptr())
        del x;gc.collect()
        np.testing.assert_array_equal(a,np.arange(16,dtype=np.float32))

    def test_spawn_transport_is_shared_and_ordered(self):
        loaders=[]
        try:
            for collate in (numpy_collate,dl._collate_fn):
                loaders.append(torch.utils.data.DataLoader(Samples(),batch_size=8,num_workers=2,
                    multiprocessing_context='spawn',persistent_workers=True,collate_fn=collate,
                    worker_init_fn=dl._WorkerInitFn(42),generator=torch.Generator().manual_seed(42)))
            old,new=iter(loaders[0]),iter(loaders[1]);retained=[]
            for i in range(4):
                a,b=next(old),next(new)
                self.assertTrue(all(t.is_shared() for t in jax.tree.leaves(b)))
                assert_bytes(self,a,b)
                np.testing.assert_array_equal(b['id'].numpy(),np.arange(i*8,(i+1)*8))
                retained.append((a,b))
            for a,b in retained:assert_bytes(self,a,b)
        finally:
            for loader in loaders:cleanup(loader)

    def test_jax_sharding_and_lifetime_after_workers_exit(self):
        mesh=jax.sharding.Mesh(np.array(jax.devices()),('data',))
        layout=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec('data'))
        loader=dl.TorchDataLoader(Samples(),8,sharding=layout,num_workers=2,num_batches=2,seed=42)
        try:
            batches=list(iter(loader));jax.block_until_ready(batches)
        finally:cleanup(loader.torch_loader)
        del loader;gc.collect()
        for i,batch in enumerate(batches):
            expected=numpy_collate([Samples()[j] for j in range(i*8,(i+1)*8)])
            reference=jax.tree.map(lambda x:jax.make_array_from_process_local_data(layout,x),expected)
            assert_bytes(self,jax.device_get(reference),jax.device_get(batch))
            for leaf in jax.tree.leaves(batch):self.assertEqual(leaf.sharding,layout)

    def test_loss_gradients_optimizer_and_ema_equal(self):
        mesh=jax.sharding.Mesh(np.array(jax.devices()),('data',))
        replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
        data=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec('data'))
        for count in (1,2,4):
            cfg,initial,_=fixture(count,0.99)
            layout=jax.tree.map(lambda x:data if getattr(x,'ndim',0)>=2 else replicated,initial)
            a=jax.device_put(jax.tree.map(lambda x:np.array(x,copy=True),initial),layout)
            a=jax.tree.map(lambda x:x.copy(),a)
            b=jax.tree.map(lambda x:x.copy(),a)
            old=[];new=[]
            for i in range(count):
                items=[Samples()[j] for j in range(i*8,(i+1)*8)]
                x,y=numpy_collate(items),dl._collate_fn(items)
                old.append(jax.tree.map(lambda v:jax.device_put(v,data),(x['features'],x['labels'])))
                new.append(jax.tree.map(lambda v:jax.device_put(dl._as_numpy(v),data),(y['features'],y['labels'])))
            engine=StreamingAccumulator(cfg,layout,data,replicated)
            def preprocess(rng,obs,**kwargs):return obs+jax.random.normal(rng,obs.shape)*0.005
            with mock.patch.object(transforms.AlohaTransformPipeline,'preprocess_observation',side_effect=preprocess):
                for _ in range(2):
                    rng=jax.random.key(42)
                    def calc(state,micros):
                        loss,grads=engine.initialize(state)
                        for i,micro in enumerate(micros):
                            loss,grads=engine.accumulate(rng,state,micro,jnp.asarray(i,jnp.int32),loss,grads)
                        return loss,grads
                    la,ga=calc(a,old);lb,gb=calc(b,new)
                    assert_bytes(self,jax.device_get((la,ga)),jax.device_get((lb,gb)))
                    a,ia=engine.update(a,la,ga);b,ib=engine.update(b,lb,gb)
                    jax.block_until_ready((a,b))
                    assert_bytes(self,jax.device_get((a,ia)),jax.device_get((b,ib)))


if __name__=='__main__':unittest.main(verbosity=2)

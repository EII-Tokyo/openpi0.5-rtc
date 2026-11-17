"""CPU checks for causal qpos memory, padding, normalization and FAST prompts."""
import unittest
from unittest.mock import patch
import numpy as np
from openpi.data.joint_position_history import JointPositionHistoryConfig, PositionHistoryBuffer, sample_positions, encode_positions
from openpi.data import transforms
from openpi.data.tokenizer import PaligemmaTokenizer
from openpi.shared.normalize import NormStats


class PositionHistoryTests(unittest.TestCase):
    def test_initial_padding_current_point_and_sampling(self):
        cfg = JointPositionHistoryConfig()
        times = np.arange(51)/50
        values = np.repeat(np.arange(51)[:,None],14,1)
        sampled = sample_positions(times,values,times[-1],cfg)
        self.assertEqual(sampled.shape,(50,14))
        np.testing.assert_array_equal(sampled[:39],0)
        np.testing.assert_array_equal(sampled[39:,0],np.arange(0,51,5))
        np.testing.assert_array_equal(sample_positions([0],values[:1],0,cfg),np.zeros((50,14)))

    def test_training_and_robot_sample_identically_then_reset(self):
        cfg=JointPositionHistoryConfig();buf=PositionHistoryBuffer(cfg)
        times=np.arange(405)/50;values=np.sin(times[:,None]+np.arange(14)[None,:])
        for i,(t,v) in enumerate(zip(times,values,strict=True)):
            buf.append(t,v)
            if i in [0,1,47,250,404]:
                np.testing.assert_array_equal(buf.sample(),sample_positions(times[:i+1],values[:i+1],t,cfg))
        buf.reset();buf.append(10,np.ones(14)*42)
        np.testing.assert_array_equal(buf.sample(),np.ones((50,14))*42)

    def test_invalid_data_never_falls_back_to_future_or_other_episode(self):
        cfg=JointPositionHistoryConfig()
        for times,values,now in [([1,0],np.zeros((2,14)),1),([0,2],np.zeros((2,14)),1),([0],np.full((1,14),np.nan),0)]:
            with self.assertRaises(ValueError):sample_positions(times,values,now,cfg)

    def test_normalization_pad_and_input_not_mutated(self):
        cfg=JointPositionHistoryConfig()
        stats=NormStats(mean=np.zeros(14),std=np.ones(14),q01=np.zeros(14),q99=np.ones(14)*2)
        raw=np.ones((50,14));result=transforms.PrepareJointPositionHistory(cfg,{'state':stats},False)({'joint_position_history':raw})
        self.assertEqual(result['joint_position_history'].shape,(50,32))
        np.testing.assert_allclose(result['joint_position_history'][:,:14],0,atol=1e-6)
        np.testing.assert_array_equal(result['joint_position_history'][:,14:],0)
        np.testing.assert_array_equal(raw,1)

    def test_fast_tokens_are_appended_once_and_never_silently_truncated(self):
        cfg=JointPositionHistoryConfig();tok=PaligemmaTokenizer(512)
        raw=np.zeros((50,32));ids=encode_positions(raw,cfg)
        tokens,mask=tok.tokenize('rinse bottle',np.zeros(14),joint_position_history=raw,joint_history_settings=cfg)
        mapped=tok._tokenizer.vocab_size()-1-128-ids
        active=tokens[mask];self.assertTrue(any(np.array_equal(active[i:i+len(mapped)],mapped) for i in range(len(active)-len(mapped)+1)))
        with self.assertRaisesRegex(ValueError,'prompt needs'):
            PaligemmaTokenizer(5).tokenize('rinse bottle',np.zeros(14),joint_position_history=raw,joint_history_settings=cfg)
        with patch('openpi.data.joint_position_history.fast_processor',return_value=lambda x:[list(range(257))]):
            with self.assertRaisesRegex(ValueError,'exceeds'):encode_positions(raw,cfg)

    def test_dataset_history_uses_original_positions_and_episode_boundaries(self):
        from openpi.data.datasets import JointPositionHistoryDataset
        class Toy:
            def __len__(self):return 12
            def __getitem__(self,i):
                return {'episode_index':i//6,'frame_index':i%6,'timestamp':(i%6)/10,'observation.state':np.ones(14)*i}
        toy=Toy();mask=np.ones(12,bool);mask[3]=False
        ds=JointPositionHistoryDataset(toy,raw_dataset=toy,settings=JointPositionHistoryConfig(),trainable_mask=mask)
        np.testing.assert_array_equal(ds[6]['joint_position_history'],np.ones((50,14))*6)
        np.testing.assert_array_equal(ds[8]['joint_position_history'][-3:,0],[6,7,8])
        with patch('openpi.data.datasets.torch.randint',return_value=__import__('torch').tensor([4])):
            result=ds[3];self.assertEqual(result['frame_index'],5)
            self.assertEqual(result['joint_position_history'][-1,0],5)

if __name__=='__main__':unittest.main()

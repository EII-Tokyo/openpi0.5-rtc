import dataclasses
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import serve_policy
from openpi.serving.checkpoint_config import checkpoint_config


class GenericEntrypointTest(unittest.TestCase):
    def test_warmup_shapes_follow_video_contract(self):
        cases=((224,6,['cam_high']), (448,6,['cam_high']), (448,2,['cam_high','cam_low','cam_left_wrist','cam_right_wrist']), (224,1,None), (224,6,None))
        cameras=['cam_high','cam_low','cam_left_wrist','cam_right_wrist']
        for size,frames,mosaic_cameras in cases:
            video=dict(camera_keys=cameras,image_resolution=[size,size],num_frames=frames,mosaic=None if mosaic_cameras is None else dict(cameras=mosaic_cameras))
            obs=serve_policy._make_dummy_obs({'runtime':{'video_input':video}})
            for camera in cameras:
                temporal=frames>1 and (mosaic_cameras is None or camera in mosaic_cameras)
                self.assertEqual(obs['images'][camera].shape,((frames,) if temporal else ())+(size,size,3))
                self.assertEqual(obs['images'][camera].dtype,np.uint8)

    def test_select_checkpoint_legacy_and_new_cli(self):
        self.assertEqual(serve_policy.selected_checkpoint(serve_policy.Args(model_dir='/x/14999')),('/x/14999',None))
        args=serve_policy.Args(policy=serve_policy.Checkpoint(config='old',dir='/x/19000'))
        self.assertEqual(serve_policy.selected_checkpoint(args),('/x/19000','old'))

    def test_manifest_rejects_corruption_and_conflicting_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'42';root.mkdir()
            file=root/'weights';file.write_bytes(b'abc')
            manifest={'config':'test','step':42,'files':[{'path':'weights','size':3,'sha256':hashlib.sha256(b'original').hexdigest()}]}
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'SHA256'):
                checkpoint_config(root)
            with self.assertRaisesRegex(ValueError,'Conflicting'):
                checkpoint_config(root,'different')
            manifest['step']=43
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'step'):
                checkpoint_config(root)
            manifest['step']=42;manifest['files'][0]['path']='../outside'
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'escapes'):
                checkpoint_config(root)


if __name__=='__main__':
    unittest.main()

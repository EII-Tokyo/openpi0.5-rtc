"""Read-only deployment checks. Live camera sampling is explicitly opt-in."""
import argparse
import time
import unittest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--model-dir')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8001)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromName('openpi.robot.aloha_real.video_input_config_test')
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
    if not args.live:
        return
    if not args.model_dir:
        parser.error('--live requires --model-dir for checkpoint validation')
    import numpy as np
    from openpi.robot.client.websocket_client_policy import WebsocketClientPolicy
    from openpi.robot.aloha_real.video_input_config import configure_video_input
    from openpi.robot.aloha_real import mosaic_runtime
    from openpi.robot.aloha_real.robot_utils import ImageRecorder
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    settings = configure_video_input(client.get_server_metadata(), args.model_dir)
    if settings['mosaic'] is not None:
        mosaic_runtime.install(settings)
    recorder = ImageRecorder(history_num_frames=settings['num_frames'],
                             history_stride_seconds=settings['stride_seconds'])
    elapsed = []
    for _ in range(300):
        started = time.monotonic()
        images = recorder.get_images()
        elapsed.append(time.monotonic()-started)
        for camera in settings['camera_keys']:
            temporal = settings['mosaic'] is None and settings['num_frames'] > 1
            temporal = temporal or (settings['mosaic'] is not None and camera in settings['mosaic']['cameras'])
            expected = ((settings['num_frames'],) if temporal else ()) + tuple(settings['image_resolution']) + (3,)
            if settings['mosaic'] is not None:
                assert images[camera].shape == expected, (camera, images[camera].shape, expected)
        time.sleep(0.005)
    p95 = np.percentile(elapsed,95)*1000
    assert p95 < 10, f'Image preparation too slow: p95={p95:.2f}ms'
    print(f'PASS checkpoint video shapes and snapshot latency p95={p95:.2f}ms')


if __name__ == '__main__':
    main()

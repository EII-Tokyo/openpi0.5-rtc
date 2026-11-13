"""Read-only ROS capture using the same checkpoint-configured adapter as runtime.
Does not instantiate a robot environment, publish actions, or dispatch a policy task.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import cv2
import numpy as np
from openpi.robot.client.websocket_client_policy import WebsocketClientPolicy
from openpi.robot.aloha_real.video_input_config import configure_video_input
from openpi.robot.aloha_real import mosaic_runtime
from openpi.robot.aloha_real.robot_utils import ImageRecorder


def save_sample(recorder, config, directory, index):
    images = recorder.get_images()
    sample_dir = directory / ('sample_%02d' % index)
    sample_dir.mkdir()
    rows = []
    details = {}
    max_columns = config['num_frames']
    height, width = config['image_resolution']
    for camera in config['camera_keys']:
        frames = np.asarray(images[camera])
        if frames.ndim == 3:
            frames = frames[None]
        details[camera] = {'shape': list(images[camera].shape)}
        row = np.full((height+32, width*max_columns, 3), 24, np.uint8)
        for i, frame in enumerate(frames):
            # Save original model-input pixels losslessly; OpenCV requires BGR.
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            path = sample_dir / ('%s_%02d.png' % (camera, i))
            if not cv2.imwrite(str(path), bgr):
                raise OSError(str(path))
            if len(frames)>1 and config['mosaic'] is not None:
                settings=config['mosaic']
                start=-settings['history_seconds']+i
                label='current' if i==len(frames)-1 else '%g to %gs' % (start, start+1-1/settings['sample_hz'])
            else:
                label='current' if len(frames)==1 else 't-%gs' % ((len(frames)-1-i)*config['stride_seconds'])
            x=i*width
            row[32:,x:x+width]=bgr
            cv2.putText(row, camera+' '+label,(x+3,21),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
        rows.append(row)
    overview = np.concatenate(rows, axis=0)
    path = directory / ('sample_%02d_overview.png' % index)
    if not cv2.imwrite(str(path), overview):
        raise OSError(str(path))
    (sample_dir/'shapes.json').write_text(json.dumps(details,indent=2))
    print(json.dumps({'overview':str(path),'shapes':details}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-dir',required=True)
    p.add_argument('--output-root',default='/app/artifacts')
    args=p.parse_args()
    metadata=WebsocketClientPolicy(host='127.0.0.1',port=8001).get_server_metadata()
    config=configure_video_input(metadata,args.model_dir)
    if config['mosaic'] is None:
        raise ValueError('This capture is for a mosaic checkpoint')
    mosaic_runtime.install(config)
    directory=Path(args.output_root)/('checkpoint%s_video_samples_' % config['checkpoint_step']+datetime.now().strftime('%Y%m%d_%H%M%S'))
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'checkpoint_video_input.json').write_text(json.dumps(config,indent=2))
    recorder=ImageRecorder(history_num_frames=config['num_frames'],history_stride_seconds=config['stride_seconds'])
    # Fill the full history using live callbacks before capturing.
    time.sleep(config['mosaic']['history_seconds']+1)
    for index in range(1,4):
        save_sample(recorder,config,directory,index)
        if index<3:
            time.sleep(2)
    print('SAVED_DIRECTORY='+str(directory),flush=True)


if __name__=='__main__':
    main()

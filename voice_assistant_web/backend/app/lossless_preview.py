"""Bounded, lossless web preview packets; no model or robot-control imports."""
import json
import struct
import time

import cv2

CAMERAS = ('cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist')


def encode_preview(frames, timestamps, sequence):
    started = time.perf_counter()
    images, rows = {}, []
    for name in CAMERAS:
        frame = frames[name]
        if frame.shape != (480, 640, 3) or str(frame.dtype) != 'uint8':
            raise ValueError('Web PNG preview requires original 640x480 uint8 camera frames')
        ok, encoded = cv2.imencode('.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise ValueError('Lossless preview PNG encoding failed')
        images[name] = encoded.tobytes()
        rows.append({'name': name, 'width': 640, 'height': 480, 'bytes': len(images[name]),
                     'received_unix': timestamps[name]})
    header = {'codec': 'png', 'sequence': sequence, 'cameras': rows,
              'encoded_unix': time.time(), 'encode_ms': (time.perf_counter() - started) * 1000}
    metadata = json.dumps(header, separators=(',', ':'), allow_nan=False).encode()
    packet = struct.pack('!I', len(metadata)) + metadata + b''.join(images.values())
    return packet, images, header

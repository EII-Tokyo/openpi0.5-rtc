import json
import struct
import unittest

import cv2
import numpy as np

from .lossless_preview import CAMERAS, encode_preview


class LosslessPreviewTest(unittest.TestCase):
    def test_all_four_original_frames_are_pixel_exact_and_metadata_matches(self):
        rng = np.random.default_rng(20261124)
        frames = {c: rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for c in CAMERAS}
        stamps = {c: 1000+i*.001 for i,c in enumerate(CAMERAS)}
        originals = {c: a.copy() for c,a in frames.items()}
        packet, pngs, header = encode_preview(frames, stamps, 5)
        length = struct.unpack('!I',packet[:4])[0]
        self.assertEqual(json.loads(packet[4:4+length]),header)
        offset = 4+length
        for row in header['cameras']:
            name = row['name'];image = packet[offset:offset+row['bytes']];offset += row['bytes']
            self.assertEqual(image,pngs[name])
            decoded = cv2.imdecode(np.frombuffer(image,np.uint8),cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(decoded,originals[name])
            np.testing.assert_array_equal(frames[name],originals[name])
            self.assertEqual(row['received_unix'],stamps[name])
        self.assertEqual(offset,len(packet))

    def test_unexpected_resolution_is_rejected_not_resized(self):
        frames = {c: np.zeros((224,224,3),np.uint8) for c in CAMERAS}
        with self.assertRaisesRegex(ValueError,'640x480'):encode_preview(frames,{c:1 for c in CAMERAS},1)


if __name__ == '__main__':
    unittest.main()

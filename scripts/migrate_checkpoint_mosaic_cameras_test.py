"""Offline migration tests: no model arrays or robot services are used."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import migrate_checkpoint_mosaic_cameras as migration


class MosaicMigrationTest(unittest.TestCase):
    def test_legacy_and_implicit_calls_become_explicit(self):
        source = "# 中文前缀\nx = T.HighCameraMosaicConfig(camera='cam_high', tile_size=112)\ny = T.HighCameraMosaicConfig(history_seconds=1)\n".encode()
        result, count = migration.migrate_source(source)
        self.assertEqual(count, 2)
        self.assertNotIn(b'camera=', result)
        self.assertIn(b"cameras=('cam_high',)", result)
        again, count = migration.migrate_source(result)
        self.assertEqual((again, count), (result, 0))

    def test_explicit_multicamera_selection_is_never_replaced(self):
        source = b"x = HighCameraMosaicConfig(camera='cam_high', cameras=('cam_high','cam_low'), history_seconds=1)"
        result, count = migration.migrate_source(source)
        self.assertEqual(count, 1)
        self.assertNotIn(b'camera=', result)
        self.assertIn(b"cameras=('cam_high','cam_low')", result)
        trailing = b"x = HighCameraMosaicConfig(cameras=('cam_high', 'cam_low'), camera='cam_high')"
        self.assertNotIn(b'camera=', migration.migrate_source(trailing)[0])

    def test_ambiguous_or_unsupported_calls_require_review(self):
        for source in [b"HighCameraMosaicConfig(camera='cam_low')", b"HighCameraMosaicConfig(**options)", b"HighCameraMosaicConfig('cam_high')"]:
            with self.assertRaises(ValueError):
                migration.migrate_source(source)

    def fixture(self, directory):
        root = Path(directory) / '42'
        snapshot = root / 'export_metadata/config.py'
        snapshot.parent.mkdir(parents=True)
        source = b"x = HighCameraMosaicConfig(camera='cam_high')\n"
        snapshot.write_bytes(source)
        manifest = {'files': [{'path': 'export_metadata/config.py', 'size': len(source),
                              'sha256': hashlib.sha256(source).hexdigest()}], 'total_bytes': len(source)}
        (root / 'manifest.json').write_text(json.dumps(manifest))
        payload = root / 'params/weights'
        payload.parent.mkdir()
        payload.write_bytes(b'unchanged tensor payload')
        return root, snapshot, payload

    def test_manifest_hashes_follow_config_and_tensor_payload_is_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root, snapshot, payload = self.fixture(directory)
            original = snapshot.read_bytes()
            info = payload.stat()
            plan = migration.plan_checkpoint(snapshot)
            report = migration.apply_plans([plan], Path(directory) / 'backup')
            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertEqual(manifest['files'][0]['sha256'], hashlib.sha256(snapshot.read_bytes()).hexdigest())
            self.assertEqual(manifest['total_bytes'], len(snapshot.read_bytes()))
            self.assertEqual(Path(report[0]['files'][0]['backup']).read_bytes(), original)
            self.assertEqual((payload.stat().st_ino, payload.stat().st_mtime_ns), (info.st_ino, info.st_mtime_ns))
            self.assertIsNone(migration.plan_checkpoint(snapshot))
            manifest['files'][0]['sha256'] = 'bad'
            snapshot.write_bytes(original)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                migration.plan_checkpoint(snapshot)

    def test_failure_rolls_back_published_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root, snapshot, _ = self.fixture(directory)
            original = snapshot.read_bytes()
            plan = migration.plan_checkpoint(snapshot)
            atomic = migration.atomic_replace
            def fail_manifest(path, content):
                if path.name == 'manifest.json':
                    raise OSError('simulated write failure')
                atomic(path, content)
            with patch.object(migration, 'atomic_replace', side_effect=fail_manifest):
                with self.assertRaises(OSError):
                    migration.apply_plans([plan], Path(directory) / 'backup')
            self.assertEqual(snapshot.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()

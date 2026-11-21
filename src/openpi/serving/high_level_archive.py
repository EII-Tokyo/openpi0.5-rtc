"""Durable local visual instruction records; never includes credentials."""
from __future__ import annotations
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid

def atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)

def atomic_json(path, value):
    atomic_bytes(Path(path), json.dumps(value, ensure_ascii=False, indent=2).encode())

class HighLevelArchive:
    def __init__(self, root):
        self.root = Path(root)
        self.session = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:12]
        self.directory = self.root / 'sessions' / self.session
        self.lock = threading.Lock()
        atomic_json(self.directory / 'session.json', {
            'schema_version': 1, 'session_id': self.session, 'created_unix': time.time(),
            'label_source': 'model_generated_not_human_verified',
            'images': 'exact decoded bytes of the images submitted to the API',
            'request_reconstruction': 'replace archive_image_sha256 with the referenced data URL',
        })

    def prepare(self, body, record):
        """Save images and exact reconstructible request BEFORE any paid call."""
        directory = self.directory / f"e{record['epoch']:04d}-r{record['seq']:08d}"
        request = copy.deepcopy(body)
        for message in request['input']:
            if not isinstance(message.get('content'), list):
                continue
            for item in message['content']:
                if item.get('type') != 'input_image':
                    continue
                prefix, encoded = item.pop('image_url').split(',', 1)
                data = base64.b64decode(encoded, validate=True)
                digest = hashlib.sha256(data).hexdigest()
                target = self.root / 'images' / digest[:2] / (digest + '.png')
                if not target.exists():
                    atomic_bytes(target, data)
                elif hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise ValueError('Existing archived image checksum mismatch')
                item['archive_image_sha256'] = digest
                item['archive_data_url_prefix'] = prefix
        wire = json.dumps(body).encode()
        atomic_json(directory / 'request.json', request)
        atomic_json(directory / 'metadata.json', {
            **record, 'session_id': self.session, 'request_wire_sha256': hashlib.sha256(wire).hexdigest(),
            'visual_input_complete': True, 'human_verified': False,
        })
        # A failed save blocks the request rather than silently losing training data.
        return str(directory)

    def finish(self, directory, record, response=None):
        directory = Path(directory)
        if response is not None:
            atomic_bytes(directory / 'response.json', response)
        atomic_json(directory / 'result.json', record)
        with self.lock:
            with (self.directory / 'index.jsonl').open('a') as f:
                f.write(json.dumps({'directory': directory.name, 'seq': record['seq'],
                    'epoch': record['epoch'], 'status': record.get('status'),
                    'cost_usd': record.get('cost_usd'), 'response_id': record.get('response_id')}) + '\n')
                f.flush()
                os.fsync(f.fileno())

    def outcome(self, directory, value):
        atomic_json(Path(directory) / 'application.json', value)

def reconstruct_request(root, directory):
    root, directory = Path(root), Path(directory)
    request = json.loads((directory / 'request.json').read_text())
    for message in request['input']:
        if not isinstance(message.get('content'), list):
            continue
        for item in message['content']:
            if 'archive_image_sha256' not in item:
                continue
            digest = item.pop('archive_image_sha256')
            prefix = item.pop('archive_data_url_prefix')
            data = (root / 'images' / digest[:2] / (digest + '.png')).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError('Archived image checksum mismatch')
            item['image_url'] = prefix + ',' + base64.b64encode(data).decode()
    # JSON member order affects the recorded wire hash. Reinsert image_url before detail.
    for message in request['input']:
        if isinstance(message.get('content'), list):
            for item in message['content']:
                if item.get('type') == 'input_image':
                    ordered = {'type': item['type'], 'image_url': item['image_url']}
                    ordered.update({k:v for k,v in item.items() if k not in ordered})
                    item.clear(); item.update(ordered)
    metadata = json.loads((directory / 'metadata.json').read_text())
    if hashlib.sha256(json.dumps(request).encode()).hexdigest() != metadata['request_wire_sha256']:
        raise ValueError('Reconstructed request checksum mismatch')
    return request

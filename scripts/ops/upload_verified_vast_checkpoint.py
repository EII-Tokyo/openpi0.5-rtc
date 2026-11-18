"""Upload a checksum-verified local Vast backup; mark complete only after S3 checks."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BUCKET = 'openpi-tokyo'
PREFIX = ('inference/eii_hxz_15repo_cam4_fullft_h200x2_448_no_mem_nearw4_5k_20261118/'
          'hxz15_h200x2_448_no_mem_bs128_ga2_5k_20261118/')


def aws(*args):
    return subprocess.check_output(['/usr/local/bin/aws', '--region', 'ap-northeast-1', *args], text=True)


def main(directory):
    root = Path(directory).resolve()
    if not (root / '.backup-complete').is_file():
        raise ValueError('Local checkpoint must pass backup checksum verification first')
    if not root.name.isdigit():
        raise ValueError('Expected a numeric checkpoint directory')
    if (root / '.s3-inference-complete').exists():
        return
    if not (root / 'params').is_dir() or not (root / 'assets').is_dir():
        raise ValueError('Inference export requires params and assets')
    files = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.name.startswith('.'):
            continue
        relative = path.relative_to(root).as_posix()
        if relative.split('/')[0] not in {'params', 'assets'} and relative not in {'inference_config.json', 'dataset_manifest.json'}:
            continue
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        key = PREFIX + root.name + '/' + relative
        # The CLI validates upload checksums; SHA-256 metadata enables restore verification.
        aws('s3', 'cp', str(path), 's3://' + BUCKET + '/' + key,
            '--only-show-errors', '--checksum-algorithm', 'SHA256', '--metadata', 'sha256=' + digest)
        head = json.loads(aws('s3api', 'head-object', '--bucket', BUCKET, '--key', key))
        if head['ContentLength'] != path.stat().st_size or head.get('Metadata', {}).get('sha256') != digest:
            raise RuntimeError('S3 verification failed for ' + relative)
        files.append({'path': relative, 'size': path.stat().st_size, 'sha256': digest})
    if not files:
        raise ValueError('Empty checkpoint')
    manifest = root.parent / ('s3-manifest-' + root.name + '.json')
    manifest.write_text(json.dumps({'type': 'inference_only', 'step': int(root.name), 'files': files}, indent=2) + '\n')
    key = PREFIX + root.name + '/_inference_manifest.json'
    aws('s3', 'cp', str(manifest), 's3://' + BUCKET + '/' + key, '--only-show-errors', '--checksum-algorithm', 'SHA256')
    remote = aws('s3', 'cp', 's3://' + BUCKET + '/' + key, '-', '--only-show-errors')
    if json.loads(remote) != json.loads(manifest.read_text()):
        raise RuntimeError('S3 completion manifest verification failed')
    (root / '.s3-inference-complete').write_text('s3://' + BUCKET + '/' + PREFIX + root.name + '/\n')


if __name__ == '__main__':
    main(sys.argv[1])

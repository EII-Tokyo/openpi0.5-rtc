import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path
import boto3
from boto3.s3.transfer import TransferConfig
from dotenv import load_dotenv

mode = sys.argv[1]
step = sys.argv[2] if len(sys.argv) > 2 else '3000'
assert step.isdigit()
repo = Path('/workspace/openpi0.5-rtc' if mode == 'upload' else '/home/eii/openpi0.5-rtc')
load_dotenv(repo / '.env')
config = 'eii_data_system_with_rinse_cam4_fullft_h200_from_april_original_30repo'
experiment = 'rinse_cam4_april_30repo_44entries_448_bs64_ga4_scan_nw96_fsdp4_fd65536_20260915'
root = repo / 'checkpoints' / config / experiment / step
prefix = f'checkpoints/{config}/{experiment}/{step}/'
s3 = boto3.client('s3', region_name='ap-northeast-1')
bucket = 'openpi-tokyo'
def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()
if mode == 'upload':
    assert json.loads((root/'_CHECKPOINT_METADATA').read_text())['commit_timestamp_nsecs']
    files = sorted(p for folder in ('params', 'assets') for p in (root/folder).rglob('*') if p.is_file())
    assert files
    def transfer(p):
        rel = p.relative_to(root).as_posix()
        sha = digest(p)
        s3.upload_file(str(p), bucket, prefix+rel, ExtraArgs={'Metadata': {'sha256': sha}}, Config=TransferConfig(max_concurrency=4))
        head = s3.head_object(Bucket=bucket, Key=prefix+rel)
        assert head['ContentLength'] == p.stat().st_size and head['Metadata']['sha256'] == sha
        print('uploaded', rel, flush=True)
        return dict(path=rel, size=p.stat().st_size, sha256=sha)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        manifest = list(pool.map(transfer, files))
    s3.put_object(Bucket=bucket, Key=prefix+'weights_manifest.json', Body=json.dumps(manifest, indent=2).encode())
else:
    assert mode == 'download'
    manifest = json.loads(s3.get_object(Bucket=bucket, Key=prefix+'weights_manifest.json')['Body'].read())
    def transfer(item):
        rel = Path(item['path'])
        assert not rel.is_absolute() and '..' not in rel.parts and rel.parts[0] in ('params', 'assets')
        p = root/rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not (p.exists() and p.stat().st_size == item['size'] and digest(p) == item['sha256']):
            tmp = p.with_name(p.name+'.download')
            s3.download_file(bucket, prefix+rel.as_posix(), str(tmp), Config=TransferConfig(max_concurrency=4))
            assert tmp.stat().st_size == item['size'] and digest(tmp) == item['sha256']
            tmp.replace(p)
        print('verified', rel, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(transfer, manifest))
    (root/'weights_manifest.json').write_text(json.dumps(manifest, indent=2))
    assert not (root/'train_state').exists()
print('COMPLETE', mode, len(manifest), sum(x['size'] for x in manifest), 's3://'+bucket+'/'+prefix, str(root.resolve()), flush=True)

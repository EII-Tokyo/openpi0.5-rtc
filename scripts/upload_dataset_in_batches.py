import argparse
import json
import os
import random
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError


def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i:i+size]


def upload_file_with_retry(api: HfApi, repo_id: str, path_in_repo: str, local_path: Path, max_retries: int, base_sleep: float):
    attempt = 0
    while True:
        try:
            api.upload_file(
                repo_id=repo_id,
                repo_type='dataset',
                path_in_repo=path_in_repo,
                path_or_fileobj=str(local_path),
            )
            return
        except HfHubHTTPError as e:
            status = getattr(e.response, 'status_code', None)
            if status != 429 or attempt >= max_retries:
                raise
            sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, base_sleep)
            print(f'429 for {path_in_repo}; sleeping {sleep_s:.1f}s before retry {attempt+1}/{max_retries}')
            time.sleep(sleep_s)
            attempt += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('repo_id')
    parser.add_argument('--local-repo-path', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sleep-between-batches', type=float, default=20.0)
    parser.add_argument('--max-retries', type=int, default=8)
    parser.add_argument('--base-sleep', type=float, default=15.0)
    parser.add_argument('--start-batch', type=int, default=0)
    args = parser.parse_args()

    local_repo = Path(args.local_repo_path)
    api = HfApi(token=os.environ.get('HF_TOKEN'))

    meta_files = []
    for rel in ['README.md', 'meta/info.json', 'meta/stats.json', 'meta/tasks.parquet', 'meta/episodes.jsonl', 'meta/episodes_stats.jsonl']:
        p = local_repo / rel
        if p.exists():
            meta_files.append((rel, p))

    for path_in_repo, local_path in meta_files:
        print('upload meta', path_in_repo)
        upload_file_with_retry(api, args.repo_id, path_in_repo, local_path, args.max_retries, args.base_sleep)

    parquet_files = sorted((local_repo / 'data').glob('*/*.parquet'))
    print('total_parquet_files', len(parquet_files))

    progress_path = local_repo / '.batch_upload_progress.json'
    progress = {
        'repo_id': args.repo_id,
        'uploaded_batches': [],
    }
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())

    for batch_idx, batch in enumerate(chunked(parquet_files, args.batch_size)):
        if batch_idx < args.start_batch or batch_idx in set(progress.get('uploaded_batches', [])):
            continue
        print(f'upload batch {batch_idx} size={len(batch)}')
        for local_path in batch:
            path_in_repo = local_path.relative_to(local_repo).as_posix()
            upload_file_with_retry(api, args.repo_id, path_in_repo, local_path, args.max_retries, args.base_sleep)
        progress.setdefault('uploaded_batches', []).append(batch_idx)
        progress_path.write_text(json.dumps(progress, indent=2))
        print(f'batch {batch_idx} done')
        time.sleep(args.sleep_between_batches)

    print('upload_done')


if __name__ == '__main__':
    main()

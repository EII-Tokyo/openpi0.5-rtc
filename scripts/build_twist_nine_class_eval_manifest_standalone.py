#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from datasets import load_dataset

CANONICAL = [
    ("Bottle on table, opening faces left", "Rotate so opening faces right"),
    ("Bottle on table, opening faces right", "Pick up with left hand"),
    ("Bottle in left hand and capped", "Unscrew cap"),
    ("Bottle in left hand, cap removed, and cap in right hand", "Bottle to left trash bin, cap to right trash bin"),
    ("Bottle in left hand, cap removed, and cap not in right hand", "Bottle to left trash bin"),
    ("Bottle in left hand and upside down", "Bottle to left trash bin"),
    ("Bottle stuck in left hand", "Use right hand to remove and place into left trash bin"),
    ("Cap on table", "Pick up cap and place into right trash bin"),
    ("No bottle on table", "Return to initial pose"),
]
REPOS = [
    "lyl472324464/2025-11-18-twist-two-bottles",
    "lyl472324464/2025-11-26-twist-two-bottles",
    "lyl472324464/2025-12-10-twist-one-bottle",
    "lyl472324464/2025-12-23-twist-one-bottle",
    "lyl472324464/2026-01-20-twist-one-bottle",
    "lyl472324464/2026-01-28-twist-many-bottle",
    "lyl472324464/2026-02-03-no-cap-and-direction",
    "lyl472324464/2026-03-04-one-direction",
    "lyl472324464/2026-03-05-two-direction",
    "lyl472324464/2026-03-09-no-cap-inference",
    "lyl472324464/2026-03-09-inference-with-and-without-cap",
    "lyl472324464/2026-03-12-one-have-cap",
    "lyl472324464/2026-03-12-one-have-cap-direction",
    "lyl472324464/2026-03-12-one-havent-cap",
    "lyl472324464/2026-03-12-one-havent-cap-direction",
    "lyl472324464/2026-03-12-two-have-all-left",
    "lyl472324464/2026-03-12-two-have-cap-all-right",
    "lyl472324464/2026-03-12-two-have-cap-one-right",
]

def class_id_for(bs, st):
    for i, (a, b) in enumerate(CANONICAL):
        if bs == a and st == b:
            return i
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--target-total', type=int, default=10000)
    ap.add_argument('--min-frame-gap', type=int, default=20)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    all_rows = []
    scanned = 0
    for repo in REPOS:
        print('Scanning', repo, flush=True)
        ds = load_dataset(
            'parquet',
            data_files=f'hf://datasets/{repo}/data/*/*.parquet',
            split='train',
        )
        subtask_col = ds.select_columns(['subtask', 'episode_index'])
        for idx in range(len(subtask_col)):
            row = subtask_col[int(idx)]
            cell = row['subtask']
            if isinstance(cell, list):
                cell = cell[0] if cell else ''
            if not isinstance(cell, str):
                cell = str(cell)
            try:
                payload = json.loads(cell)
            except Exception:
                continue
            bs = payload.get('bottle_state') or payload.get('Bottle State')
            st = payload.get('subtask') or payload.get('Subtask')
            if not bs or not st:
                continue
            cid = class_id_for(bs, st)
            if cid is None:
                continue
            ep = int(row['episode_index'])
            all_rows.append({
                'repo_id': repo,
                'frame_index': int(idx),
                'episode_index': ep,
                'class_id': cid,
                'bottle_state': bs,
                'subtask': st,
            })
            scanned += 1
    if len(all_rows) < args.target_total:
        raise RuntimeError(f'only {len(all_rows)} canonical rows < target {args.target_total}')

    raw_class_counts = Counter(int(r['class_id']) for r in all_rows)
    order = rng.permutation(len(all_rows))
    selected = []
    accepted_frames = defaultdict(list)
    for i in order:
        row = all_rows[int(i)]
        key = (row['repo_id'], int(row['episode_index']))
        frame_index = int(row['frame_index'])
        prev = accepted_frames[key]
        if prev and min(abs(frame_index - x) for x in prev) < args.min_frame_gap:
            continue
        selected.append(row)
        prev.append(frame_index)
        if len(selected) >= args.target_total:
            break

    if len(selected) < args.target_total:
        raise RuntimeError(
            f'only {len(selected)} rows satisfy min_frame_gap={args.min_frame_gap} < target {args.target_total}'
        )

    selected.sort(key=lambda r: (r['class_id'], r['repo_id'], r['episode_index'], r['frame_index']))
    selected_class_counts = Counter(int(r['class_id']) for r in selected)

    payload = {
        'summary': {
            'num_repos': len(REPOS),
            'scanned_canonical_rows': scanned,
            'selected_rows': len(selected),
            'min_frame_gap': args.min_frame_gap,
            'available_raw_per_class': {cid: raw_class_counts.get(cid, 0) for cid in range(9)},
            'selected_per_class_counts': {cid: selected_class_counts.get(cid, 0) for cid in range(9)},
        },
        'canonical_pairs': [
            {'class_id': i, 'bottle_state': bs, 'subtask': st} for i, (bs, st) in enumerate(CANONICAL)
        ],
        'rows': selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload['summary'], ensure_ascii=False, indent=2))
    print('Saved manifest to:', args.output)

if __name__ == '__main__':
    main()

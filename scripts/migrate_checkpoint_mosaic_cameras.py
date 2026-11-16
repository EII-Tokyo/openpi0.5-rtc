"""One-time explicit migration of exported configs, never imported by inference."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def migrate_source(content):
    """Rewrite only mosaic constructor arguments, retaining their camera selection."""
    tree = ast.parse(content)
    lines = content.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    def position(node, end=False):
        return starts[(node.end_lineno if end else node.lineno) - 1] + (node.end_col_offset if end else node.col_offset)
    edits = []
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node.func, 'attr', getattr(node.func, 'id', None)) != 'HighCameraMosaicConfig':
            continue
        if node.args or any(kw.arg is None for kw in node.keywords):
            raise ValueError('Positional or unpacked mosaic arguments need explicit review')
        keywords = {kw.arg: kw for kw in node.keywords}
        old = keywords.get('camera')
        if old is not None:
            if ast.literal_eval(old.value) != 'cam_high':
                raise ValueError('Only the previously supported cam_high selector can be migrated')
            start, end = position(old), position(old, True)
            if 'cameras' in keywords:
                after = end
                while after < len(content) and content[after:after+1].isspace():
                    after += 1
                if content[after:after+1] == b',':
                    end = after + 1
                else:
                    before = start - 1
                    while before >= 0 and content[before:before+1].isspace():
                        before -= 1
                    if content[before:before+1] != b',':
                        raise ValueError('Cannot safely remove the old keyword separator')
                    start = before
                edits.append((start, end, b''))
            else:
                edits.append((start, end, b"cameras=('cam_high',)"))
            count += 1
        elif 'cameras' not in keywords:
            opening = content.index(b'(', position(node.func, True)) + 1
            edits.append((opening, opening, b"cameras=('cam_high',),"))
            count += 1
    result = content
    for start, end, replacement in sorted(edits, reverse=True):
        result = result[:start] + replacement + result[end:]
    ast.parse(result)
    return result, count


def snapshots(root):
    """Include checkpoint aliases once and avoid walking tensor payloads."""
    seen = set()
    for base, directories, files in os.walk(root, followlinks=True):
        real = Path(base).resolve()
        if real in seen:
            directories[:] = []
            continue
        seen.add(real)
        directories[:] = [name for name in directories if name not in {'params', 'train_state', 'assets', '.git', '__pycache__'}]
        if Path(base).name == 'export_metadata' and 'config.py' in files:
            yield (Path(base) / 'config.py').resolve()


def plan_checkpoint(snapshot):
    """Verify existing config hashes before planning any metadata replacement."""
    before = snapshot.read_bytes()
    after, count = migrate_source(before)
    if before == after:
        return None
    root = snapshot.parent.parent
    changes = {snapshot: (before, after)}
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        original = manifest_path.read_bytes()
        manifest = json.loads(original)
        difference = 0
        for entry in manifest.get('files', []):
            if entry.get('path') == 'export_metadata/config.py':
                if entry.get('size') != len(before) or entry.get('sha256') != sha256(before):
                    raise ValueError(f'Existing config manifest mismatch: {snapshot}')
                entry.update(size=len(after), sha256=sha256(after))
                difference += len(after) - len(before)
        if 'total_bytes' in manifest:
            manifest['total_bytes'] += difference
        changes[manifest_path] = (original, (json.dumps(manifest, indent=2) + '\n').encode())
    return {'root': root, 'constructors': count, 'changes': changes}


def atomic_replace(path, content):
    """Publish one complete metadata file and preserve its owner and permissions."""
    info = path.stat()
    fd, temporary = tempfile.mkstemp(prefix='.mosaic-migration-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            os.fchmod(stream.fileno(), info.st_mode & 0o777)
            if (os.fstat(stream.fileno()).st_uid, os.fstat(stream.fileno()).st_gid) != (info.st_uid, info.st_gid):
                os.fchown(stream.fileno(), info.st_uid, info.st_gid)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def apply_plans(plans, backup):
    """Back up all metadata before writing; restore every published file on failure."""
    backup.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = []
    for index, plan in enumerate(plans):
        directory = backup / str(index)
        directory.mkdir()
        rows = []
        for number, (path, (before, after)) in enumerate(plan['changes'].items()):
            if path.read_bytes() != before:
                raise ValueError(f'Config changed since planning: {path}')
            saved = directory / str(number)
            saved.write_bytes(before)
            rows.append({'path': str(path), 'backup': str(saved), 'before_sha256': sha256(before),
                         'after_sha256': sha256(after), 'before_bytes': len(before), 'after_bytes': len(after)})
        report.append({'checkpoint': str(plan['root']), 'constructors': plan['constructors'], 'files': rows})
    (backup / 'migration.json').write_text(json.dumps(report, indent=2) + '\n')
    published = []
    try:
        for plan in plans:
            for path, (before, after) in plan['changes'].items():
                if path.read_bytes() != before:
                    raise ValueError(f'Config changed during migration: {path}')
                atomic_replace(path, after)
                published.append((path, before))
    except BaseException:
        for path, before in reversed(published):
            atomic_replace(path, before)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--backup-dir', type=Path)
    args = parser.parse_args()
    if args.apply and args.backup_dir is None:
        parser.error('--apply requires --backup-dir outside the checkpoint tree')
    root = args.checkpoint_root.resolve()
    if args.backup_dir and args.backup_dir.resolve().is_relative_to(root):
        parser.error('Keep backups outside the checkpoint tree')
    plans = [plan for snapshot in snapshots(root) if (plan := plan_checkpoint(snapshot))]
    report = apply_plans(plans, args.backup_dir) if args.apply and plans else []
    print(json.dumps({'checkpoints': len(plans), 'constructors': sum(plan['constructors'] for plan in plans),
                      'applied': bool(args.apply), 'backup': str(args.backup_dir) if report else None}, indent=2))


if __name__ == '__main__':
    main()

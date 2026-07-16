from pathlib import Path
import json
import FreeCAD

FCSTD = Path('/home/eii/project/openpi0.5-rtc-reward-learning/scene_reconstruction/cad/aloha_incremental/iterations/iter_003_lower_camera_top_position/iter_003_lower_camera_top_position.FCStd')
OUT = Path('/home/eii/project/openpi0.5-rtc-reward-learning/scene_reconstruction/cad/aloha_incremental/diagnostics/iter003_mesh_diagnostics.json')

doc = FreeCAD.openDocument(str(FCSTD))
rows = []
for obj in doc.Objects:
    if not hasattr(obj, 'Mesh'):
        continue
    mesh = obj.Mesh
    row = {'name': obj.Name, 'label': getattr(obj, 'Label', ''), 'source_asset': getattr(obj, 'SourceAsset', None)}
    for attr in ['CountFacets', 'CountPoints', 'CountEdges']:
        try:
            row[attr] = int(getattr(mesh, attr))
        except Exception as exc:
            row[attr] = f'ERR:{exc}'
    checks = {}
    for method in [
        'isSolid',
        'hasNonManifolds',
        'hasSelfIntersections',
        'hasNonUniformOrientedFacets',
        'countNonUniformOrientedFacets',
        'countComponents',
        'countSegments',
    ]:
        try:
            fn = getattr(mesh, method)
        except Exception as exc:
            checks[method] = f'MISSING:{exc}'
            continue
        try:
            checks[method] = fn()
        except Exception as exc:
            checks[method] = f'ERR:{type(exc).__name__}:{exc}'
    row['checks'] = checks
    # Dry-run repair on a copy and report count deltas, do not modify document.
    try:
        m = mesh.copy()
        before = (int(m.CountPoints), int(m.CountFacets))
        repair_steps = []
        for method in ['fixIndices', 'removeDuplicatedPoints', 'removeDuplicatedFacets', 'fixDegenerations', 'harmonizeNormals']:
            if hasattr(m, method):
                try:
                    ret = getattr(m, method)()
                    repair_steps.append([method, str(ret)])
                except Exception as exc:
                    repair_steps.append([method, f'ERR:{type(exc).__name__}:{exc}'])
        after = (int(m.CountPoints), int(m.CountFacets))
        row['repair_dry_run'] = {'before_points_facets': before, 'after_points_facets': after, 'steps': repair_steps}
    except Exception as exc:
        row['repair_dry_run'] = f'ERR:{type(exc).__name__}:{exc}'
    rows.append(row)

OUT.write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
print(OUT)
for row in rows:
    c = row['checks']
    bad = []
    for k, v in c.items():
        if isinstance(v, bool) and v and k.startswith('has'):
            bad.append(f'{k}={v}')
        elif k.startswith('count') and isinstance(v, int) and v > 0 and k != 'countComponents':
            bad.append(f'{k}={v}')
    dry = row.get('repair_dry_run', {})
    changed = isinstance(dry, dict) and dry.get('before_points_facets') != dry.get('after_points_facets')
    if bad or changed:
        print(row['name'], 'source=', row.get('source_asset'), 'bad=', bad, 'dry=', dry)

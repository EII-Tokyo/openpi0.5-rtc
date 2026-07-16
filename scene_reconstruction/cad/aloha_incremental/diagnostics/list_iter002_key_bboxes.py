import json
from pathlib import Path
import FreeCAD as App
path=Path('/home/eii/project/openpi0.5-rtc-reward-learning/scene_reconstruction/cad/aloha_incremental/iterations/iter_002_measured_aloha_y_offset/iter_002_measured_aloha_y_offset.FCStd')
doc=App.openDocument(str(path))
rows=[]
for obj in doc.Objects:
    box=None
    if hasattr(obj,'Mesh'):
        box=obj.Mesh.BoundBox
    elif hasattr(obj,'Shape'):
        box=obj.Shape.BoundBox
    if box:
        rows.append({
            'name':obj.Name,'label':obj.Label,'type':obj.TypeId,
            'bbox':[box.XMin,box.XMax,box.YMin,box.YMax,box.ZMin,box.ZMax],
            'center':[(box.XMin+box.XMax)/2,(box.YMin+box.YMax)/2,(box.ZMin+box.ZMax)/2],
            'size':[box.XLength,box.YLength,box.ZLength],
        })
App.closeDocument(doc.Name)
Path('scene_reconstruction/cad/aloha_incremental/diagnostics/iter002_key_bboxes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
for r in rows:
    name=r['name']
    if any(s in name for s in ['REF_TABLE','REF_SCENE_frame_extrusion_1220','REF_SCENE_frame_extrusion_1000','REF_SCENE_frame_wormseye','REF_SCENE_frame_extrusion_600','REF_SCENE_frame_extrusion_150_']):
        print(name, 'center', [round(x,1) for x in r['center']], 'size', [round(x,1) for x in r['size']], 'bbox', [round(x,1) for x in r['bbox']])

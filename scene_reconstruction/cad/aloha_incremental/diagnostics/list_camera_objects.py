import json
from pathlib import Path
import FreeCAD as App
ROOT=Path('/home/eii/project/openpi0.5-rtc-reward-learning')
paths={
 'iter002': ROOT/'scene_reconstruction/cad/aloha_incremental/iterations/iter_002_measured_aloha_y_offset/iter_002_measured_aloha_y_offset.FCStd',
 'iter003': ROOT/'scene_reconstruction/cad/aloha_incremental/iterations/iter_003_lower_camera_top_position/iter_003_lower_camera_top_position.FCStd',
}
out={}
for key,path in paths.items():
    doc=App.openDocument(str(path))
    rows=[]
    for obj in doc.Objects:
        name=(obj.Name+' '+obj.Label).lower()
        if any(s in name for s in ['camera','cam','worm','mount','frame']):
            box=None
            if hasattr(obj,'Mesh'):
                box=obj.Mesh.BoundBox
            elif hasattr(obj,'Shape'):
                box=obj.Shape.BoundBox
            if box:
                rows.append({
                    'name': obj.Name, 'label': obj.Label, 'type': obj.TypeId,
                    'bbox': [box.XMin,box.XMax,box.YMin,box.YMax,box.ZMin,box.ZMax],
                    'center': [(box.XMin+box.XMax)/2,(box.YMin+box.YMax)/2,(box.ZMin+box.ZMax)/2],
                    'size': [box.XLength, box.YLength, box.ZLength],
                })
            else:
                rows.append({'name':obj.Name,'label':obj.Label,'type':obj.TypeId,'bbox':None})
    out[key]=rows
    App.closeDocument(doc.Name)
Path('scene_reconstruction/cad/aloha_incremental/diagnostics/camera_objects_iter002_iter003.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2)[:12000])

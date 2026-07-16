import json
from pathlib import Path
import FreeCAD as App
p=Path('/home/eii/project/openpi0.5-rtc-reward-learning/scene_reconstruction/cad/aloha_incremental/iterations/iter_003_lower_camera_top_position/iter_003_lower_camera_top_position.FCStd')
doc=App.openDocument(str(p))
result={}
for name in ['REF_SCENE_frame_wormseye_mount_30','NEW_LOWER_CAMERA_POSITION_GREEN','MEASURED_TOP_STEEL_RAIL_DARK_REFERENCE'] + [f'MEASURED_CAMERA_SUPPORT_PIPE_260MM_{i}' for i in range(1,5)]:
    obj=doc.getObject(name)
    if obj is None:
        result[name]=None
        continue
    box=obj.Mesh.BoundBox if hasattr(obj,'Mesh') else obj.Shape.BoundBox
    result[name]={
        'type': obj.TypeId,
        'center': [(box.XMin+box.XMax)/2,(box.YMin+box.YMax)/2,(box.ZMin+box.ZMax)/2],
        'size': [box.XLength, box.YLength, box.ZLength],
    }
App.closeDocument(doc.Name)
out=Path('scene_reconstruction/cad/aloha_incremental/diagnostics/verify_iter003_corrected.json')
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))

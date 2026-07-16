import FreeCAD
from pathlib import Path
fcstd=Path('/home/eii/project/openpi0.5-rtc-reward-learning/scene_reconstruction/cad/aloha_incremental/iterations/iter_001_measured_table_rack/iter_001_measured_table_rack.FCStd')
doc=FreeCAD.openDocument(str(fcstd))
for obj in doc.Objects:
    if obj.Name.startswith('REF_ALOHA_right_base_link_vx300s_1_base_0'):
        bb=obj.Mesh.BoundBox
        print('object', obj.Name)
        print('bbox', bb.XMin, bb.XMax, bb.YMin, bb.YMax, bb.ZMin, bb.ZMax)
        print('lengths', bb.XLength, bb.YLength, bb.ZLength)
        print('center_y', (bb.YMin+bb.YMax)/2)
        break
else:
    print('not found')

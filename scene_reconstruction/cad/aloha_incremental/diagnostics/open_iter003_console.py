import FreeCAD as App
path='/home/eii/project/openpi0.5-rtc-reward-learning/scene_reconstruction/cad/aloha_incremental/iterations/iter_003_lower_camera_top_position/iter_003_lower_camera_top_position.FCStd'
doc=App.openDocument(path)
print('opened', doc.Name, 'objects', len(doc.Objects))
App.closeDocument(doc.Name)

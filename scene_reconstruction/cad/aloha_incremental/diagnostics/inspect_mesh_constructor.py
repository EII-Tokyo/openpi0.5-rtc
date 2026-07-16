import Mesh
print('Mesh module', Mesh)
print('Mesh.Mesh doc:', getattr(Mesh.Mesh, '__doc__', None))
print('Mesh attrs containing create/topo')
for name in dir(Mesh):
    if any(s in name.lower() for s in ['mesh','topo','create','facet']):
        print(name)

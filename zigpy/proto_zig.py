import zigpy.mzig.octree as zn

print(dir(zn))
domain_size = (2, 4)
periodic_dim = (False, True)

a = zn.PyForest2D(domain_size, periodic_dim)

print(a)
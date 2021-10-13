import lattice
import numpy as np


def test_neighbor_list():
    lat = lattice.Lattice('triangle', 3, 3, unit_cell_mult=1)
    neighbor_list = lat.get_neighbor_list(distance_index=0)


origin = np.array([0,0])
a1 = np.array([1j,0])
a2 = np.array([0,1])
Lx = 3
Ly = 4

Tmat = lattice.get_translations(a1, a2, Lx, Ly)
print(Tmat[:,0,2])
dist = Tmat - a1[:,None,None]
distances = np.linalg.norm(dist,axis=0)
print(distances)
print(np.argmin(distances))
print(np.unravel_index(np.argmin(distances), distances.shape))
print(lattice.shortest_distance(a1,a2,Tmat))
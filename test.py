import lattice
import montecarlo as mc
import numpy as np

lat = lattice.Lattice(lat_type='triangle', lx=2, ly=2)
print(lat.basis)

lat = lattice.Lattice(lat_type='triangle', lx=2, ly=3, unit_cell_mult=3)
print(lat.basis)
print(lat.a1)
print(lat.a2)
print(lat.coordinates)
print(lat.distances)
print(lat.neighbor_table[0])
print(lat.neighbor_pbc[0])
print(lat.get_neighbor_pairs(0))

conf_init = {
    'size': lat.N,
    'S2': 2,
    'num_each': (6, 6, 6)
}
MC = mc.MonteCarlo(conf_init, {2: lat.get_neighbor_pairs(0)})
print(MC.neighbor_list)
print(MC.propose_move())
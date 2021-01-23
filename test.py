import lattice

lat = lattice.Lattice(lat_type='triangle', lx=2, ly=2)
print(lat.basis)

lat = lattice.Lattice(lat_type='triangle', lx=2, ly=3, unit_cell_mult=3)
print(lat.basis)
print(lat.a1)
print(lat.a2)
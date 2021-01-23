import numpy as np


def get_lattice_geometry(lattice_type, basis=):
    if lattice_type == 'triangle':


class Lattice:
    def __init__(self, lat_type, lx, ly, unit_cell_mult=1):
        if lat_type == 'triangle':
            x = np.array([1.0, 0.0])
            self.basis = x[:, None] * np.arange(unit_cell_mult)[None, :]
            self.a1 = x*unit_cell_mult
            self.a2 = np.array([0.5, 0.5*np.sqrt(3)])
            rlist = []
            for i in np.arange(lx):
                for j in np.arange(ly):
                    for b in np.arange(unit_cell_mult):
                        rlist.append(self.a1*i + self.a2*j + self.basis[:, b])

            self.coordinates = np.array(rlist).transpose()
import numpy as np

class Lattice:
    def __init__(self, lat_type, lx, ly, unit_cell_mult=1):
        if lat_type == 'triangle':
            x = np.array([1.0, 0.0])
            self.basis = x[:, None] * np.arange(unit_cell_mult)[None, :]
            basemat = np.dstack([self.basis]*lx*ly).transpose((0, 2, 1)).reshape((2, lx*ly*unit_cell_mult))
            print(basemat)

            self.a1 = x*unit_cell_mult
            self.a2 = np.array([0.5, 0.5*np.sqrt(3)])
            xmat = np.concatenate([np.arange(lx)]*ly).reshape((ly, 2)).transpose()
            ymat = np.dstack([np.arange(ly)]*lx).transpose()
            print(xmat)
            print(ymat)

            # self.coordinates = np.repeat(self.a1[np.newaxis, :] * i, unit_cell_mult, axis=0) + self.basis
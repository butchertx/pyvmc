import numpy as np


def get_lattice_geometry(lattice_type, lx, ly, unit_cell_mult=1):
    if lattice_type == 'triangle':
        x = np.array([1.0, 0.0])
        basis = x[:, None] * np.arange(unit_cell_mult)[None, :]
        a1 = x*unit_cell_mult
        a2 = np.array([0.5, 0.5*np.sqrt(3)])
        rlist = []
        for i in np.arange(lx):
            for j in np.arange(ly):
                for b in np.arange(unit_cell_mult):
                    rlist.append(a1 * i + a2 * j + basis[:, b])

        coordinates = np.array(rlist).transpose()
        return basis, a1, a2, coordinates


def get_translations(a1, a2, lx, ly):
    # Tmat[:,i,j] gives the lattice translation for (i-1)*T1 + (j-1)*T2
    T1, T2 = np.matmul(a1[:, None], np.array([[-1, 0, 1], ]))*lx, np.matmul(a2[:, None], np.array([[-1, 0, 1], ]))*ly
    return T1[:, :, None] + T2[:, None, :]


def shortest_distance(x2, Tmat, x1=None):
    # calculate the shortest distance using Tmat and return the distance, T/F Lx translation, T/F Ly translation
    if x1 is None:
        diffs = x2[:, None, None] + Tmat
    else:
        diffs = x2[:, None, None] - x1[:, None, None] + Tmat

    distances = np.linalg.norm(diffs, axis=0)
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
    return distances[min_idx], min_idx[0] != 1, min_idx[1] != 1


def get_distances(coordinates, Tmat):
    distances = []
    for x in coordinates.transpose():
        distances.append(shortest_distance(x, Tmat)[0])

    return np.sort(list(set(np.round(distances, decimals=10))), axis=0)


def get_neighbor_table(coordinates, N, distances, Tmat, max_dist=3):
    neighbors = []
    neighbor_pbc = []
    atol = 1e-8  # Absolute tolerance
    for x1, idx1 in zip(coordinates.transpose(), np.arange(N)):
        neighbors.append([])
        neighbor_pbc.append([])
        for i in distances[:max_dist]:
            neighbors[idx1].append([])
            neighbor_pbc[idx1].append([])

        for x2, idx2 in zip(coordinates.transpose(), np.arange(N)):
            if idx2 != idx1:
                dist, lx_pbc, ly_pbc = shortest_distance(x2, Tmat, x1=x1)
                distance_index = np.where(abs(dist - distances) < atol)[0][0]
                if distance_index < max_dist:
                    neighbors[idx1][distance_index].append(idx2)
                    neighbor_pbc[idx1][distance_index].append(1)

    return neighbors, neighbor_pbc


class Lattice:
    def __init__(self, lat_type, lx, ly, unit_cell_mult=1):
        self.basis, self.a1, self.a2, self.coordinates = get_lattice_geometry(lat_type, lx, ly, unit_cell_mult)
        self.N = self.basis.shape[1] * lx * ly
        self.Tmat = get_translations(self.a1, self.a2, lx, ly)
        self.distances = get_distances(self.coordinates, self.Tmat)[1:]
        self.neighbor_table, self.neighbor_pbc = get_neighbor_table(self.coordinates, self.N, self.distances, self.Tmat)




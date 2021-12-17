import numpy as np

LATTICE_TOL = 1e-10


def compare_position(x1, x2):
    return (np.abs(x1-x2) < LATTICE_TOL).all()


def get_lattice_geometry(lattice_type, lx, ly, unit_cell_mult=1):
    if lattice_type == 'triangle':
        x = np.array([1.0, 0.0])
        basis = x[:, None] * np.arange(unit_cell_mult)[None, :]
        a1 = x*unit_cell_mult
        a2 = np.array([0.5, 0.5*np.sqrt(3)])
    elif lattice_type == 'square':
        x = np.array([1.0, 0.0])
        basis = x[:, None] * np.arange(unit_cell_mult)[None, :]
        a1 = x * unit_cell_mult
        a2 = np.array([0.0, 1.0])
    else:
        raise NotImplementedError(f'Lattice type {lattice_type} not implemented in lattice.py')

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


def get_displacement_(x2, Tmat, x1=None):
    # get the shortest displacement vector from x1 to x2 using periodic translations
    # return displacement vector, distance, T/F Lx translation, T/F Ly translation
    if x1 is None:
        diffs = x2[:, None, None] + Tmat
    else:
        diffs = x2[:, None, None] - x1[:, None, None] + Tmat

    distances = np.linalg.norm(diffs, axis=0)
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
    return diffs[:, min_idx[0], min_idx[1]], distances[min_idx], min_idx[0] != 1, min_idx[1] != 1


def get_displacement(x2, Tmat, x1=None):
    # get the shortest displacement vector from x1 to x2 using periodic translations
    # return only the displacement
    diff, distance, Lx_trans, Ly_trans = get_displacement_(x2, Tmat, x1=x1)
    return diff


def shortest_distance(x2, Tmat, x1=None):
    # calculate the shortest distance using Tmat and return the distance, T/F Lx translation, T/F Ly translation
    diff, distance, Lx_trans, Ly_trans = get_displacement_(x2, Tmat, x1=x1)
    return distance, Lx_trans, Ly_trans


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


def get_triangle_ring_list(coordinates, neighbor_list, Tmat):
    """
    Algorithm: 1.  For each index site, get the list of nearest neighbors
    2.  Select the sites that are an a1, a2 and a2-a1 translation away from the index site
    3.  append to list of rings the two triangles made of the index site and the three neighbor sites
    :param neighbor_list:
    :return: list of tuples: each tuple is a ring exchange triple, ordered counterclockwise.  Two rings per site
    """
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, 0.5 * np.sqrt(3)])
    a2_minus_a1 = a2 - a1
    ring_list = []
    for site0 in np.arange(coordinates.shape[1]):
        lower_site = None
        diag_site = None
        upper_site = None
        for candidate in neighbor_list[site0, :]:
            if compare_position(a1, get_displacement(coordinates[:, candidate], Tmat, coordinates[:, site0])):
                lower_site = candidate
            elif compare_position(a2, get_displacement(coordinates[:, candidate], Tmat, coordinates[:, site0])):
                diag_site = candidate
            elif compare_position(a2_minus_a1, get_displacement(coordinates[:, candidate], Tmat, coordinates[:, site0])):
                upper_site = candidate
        if (diag_site is None) or (lower_site is None) or (upper_site is None):
            raise RuntimeError('No sites match triangle lattice displacement for ring exchange')

        ring_list += [(site0, lower_site, diag_site), (site0, diag_site, upper_site)]

    return ring_list


class Lattice:
    def __init__(self, lat_type, lx, ly, unit_cell_mult=1):
        self.lat_type = lat_type
        self.basis, self.a1, self.a2, self.coordinates = get_lattice_geometry(lat_type, lx, ly, unit_cell_mult)
        self.N = self.basis.shape[1] * lx * ly
        self.Tmat = get_translations(self.a1, self.a2, lx, ly)
        self.distances = get_distances(self.coordinates, self.Tmat)[1:]
        self.neighbor_table, self.neighbor_pbc = get_neighbor_table(self.coordinates, self.N, self.distances, self.Tmat)

    def get_neighbor_pairs(self, distance_index=2):
        """
        :param distance_index: distance between neighbors
        :return: list of all site pairs the specified distance apart
        """
        neighbor_dist = self.get_neighbor_list(distance_index) # np.array([neigh[distance_index] for neigh in self.neighbor_table])
        tpl_list = [(i, j) for i in range(len(neighbor_dist)) for j in neighbor_dist[i]]
        return tpl_list

    def get_neighbor_list(self, distance_index=2):
        """
        get list of lists of neighbors at distance "distance_index"
        :param distance_index: distance between neighbors
        :return: np.array: [ [n1, n2, n3...], [m1, m2, m3,...]...] where row = site index, col = neighbor index
        """
        return np.array([neigh[distance_index] for neigh in self.neighbor_table])


class TriangleLattice(Lattice):
    def __init__(self, lat_type, lx, ly, unit_cell_mult=1):
        """
        Triangle Lattice
        same as Lattice with an extra ring exchange method
        :param same input params as Lattice object
        """
        Lattice.__init__(self, lat_type, lx, ly, unit_cell_mult=unit_cell_mult)
        assert (self.lat_type == 'triangle')

    def get_ring_exchange_list(self):
        return get_triangle_ring_list(self.coordinates, self.get_neighbor_list(distance_index=0), self.Tmat)

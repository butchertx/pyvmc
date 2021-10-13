import lattice
import wavefunction
import local_operator
import montecarlo
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)


UCX = 2
UCY = 2


def afq3_triangle(uc_x, uc_y):
    directors_uc1 = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
    stack1 = np.tile(directors_uc1, (1, uc_y))
    directors_uc2 = np.array([[0, 1, 0],
                              [0, 0, 1],
                              [1, 0, 0]])
    stack2 = np.tile(directors_uc2, (1, uc_y))
    directors_uc3 = np.array([[0, 0, 1],
                              [1, 0, 0],
                              [0, 1, 0]])
    stack3 = np.tile(directors_uc3, (1, uc_y))
    uc_filled = np.concatenate([stack1, stack2, stack3], axis=1)
    directors = np.tile(uc_filled, (1, uc_x))
    return directors


def afm120_triangle(uc_x, uc_y):
    directors_uc1 = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
    stack1 = np.tile(directors_uc1, (1, uc_y))
    directors_uc2 = np.array([[0, 1, 0],
                              [0, 0, 1],
                              [1, 0, 0]])
    stack2 = np.tile(directors_uc2, (1, uc_y))
    directors_uc3 = np.array([[0, 0, 1],
                              [1, 0, 0],
                              [0, 1, 0]])
    stack3 = np.tile(directors_uc3, (1, uc_y))
    uc_filled = np.concatenate([stack1, stack2, stack3], axis=1)
    directors = np.tile(uc_filled, (1, uc_x))
    return directors


def create_lattice():
    return lattice.Lattice('triangle', 3*UCY, 3*UCX, unit_cell_mult=1)


def create_wavefunction(lattice_in):
    conf_init = {
        'size': lattice_in.N,
        'S2': 2,
        'num_each': (3*UCX*UCY, 3*UCX*UCY, 3*UCX*UCY)
    }
    directors_u = afq3_triangle(UCX, UCY)
    d_rotate = wavefunction.euler_s1(-3.0 * np.pi / 4.0, np.arccos(np.sqrt(1.0 / 3.0)), 3.0 * np.pi / 4.0, directors_u,
                                     sz_basis=False)
    afq3 = wavefunction.ProductState(conf_init, d_rotate)
    return afq3


def create_hamiltonian(lattice_in):
    H = local_operator.Hamiltonian()
    H.add_term('J', 1.0, lattice_in.get_neighbor_pairs(0))
    # H.add_term('K', 1.0, lattice_in.get_neighbor_pairs(0))
    return H


lattice_run = create_lattice()
wavefunction_run = create_wavefunction(lattice_run)
ham_run = create_hamiltonian(lattice_run)

mc = montecarlo.MonteCarlo(wavefunction_run, lattice_run.get_neighbor_pairs(0), measures=1000)
mc.add_observable(ham_run)
results, per_site, measurements = mc.run()
print(per_site)
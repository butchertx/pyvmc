import lattice
import wavefunction
import local_operator
import montecarlo
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)


UCX = 3
UCY = 3


def neel_directors(uc_x, uc_y):
    normu = 1.0 / np.sqrt(2.0)
    directors_uc1 = np.array([[normu, -normu],
                              [0, 0],
                              [0, 0]])
    stack1 = np.tile(directors_uc1, (1, uc_y))
    directors_uc2 = np.array([[-normu, normu],
                              [0, 0],
                              [0, 0]])
    stack2 = np.tile(directors_uc2, (1, uc_y))
    uc_filled = np.concatenate([stack1, stack2], axis=1)
    directors_u = np.tile(uc_filled, (1, uc_x))

    directors_uc = np.array([[0, 0],
                             [0, 0],
                             [normu, normu]])
    stack1 = np.tile(directors_uc, (1, uc_y))
    uc_filled = np.concatenate([stack1, stack1], axis=1)
    directors_v = np.tile(uc_filled, (1, uc_x))
    return directors_u, directors_v


def create_lattice():
    return lattice.Lattice('square', 2*UCY, 2*UCX, unit_cell_mult=1)


def create_wavefunction(lattice_in, state_type='neel'):
    conf_init = {
        'size': lattice_in.N,
        'S2': 2,
        'num_each': (int(4*UCX*UCY/3), int(4*UCX*UCY/3), int(4*UCX*UCY/3))
    }
    jastrow_init = [
        {
            'couples_to': 'sz',
            'strength': -0.1,
            'neighbors': lattice_in.get_neighbor_list(distance_index=j)
        }
        for j in range(1)
    ]
    if state_type == 'neel':
        directors_u, directors_v = neel_directors(UCX, UCY)
        directors_d = directors_u + 1j*directors_v
        neel = wavefunction.ProductState(conf_init, directors_d, jastrow_init)
        return neel
    else:
        print('No valid state selected')
        exit()


def create_hamiltonian(lattice_in):
    neighbor_pairs = list(set(map(tuple, map(sorted, lattice_in.get_neighbor_pairs(0))))) #  remove duplicate neighbor pairs
    H = local_operator.Hamiltonian()
    H.add_term('J', 1.0, neighbor_pairs, interaction_type=local_operator.HeisenbergExchange)
    return H


if __name__ == '__main__':
    lattice_run = create_lattice()
    wavefunction_run = create_wavefunction(lattice_run)
    ham_run = create_hamiltonian(lattice_run)
    mc = montecarlo.MonteCarlo(wavefunction_run, lattice_run.get_neighbor_pairs(0), measures=1000)
    mc.add_observable(ham_run)
    results, per_site, measurements = mc.run()
    print(per_site)
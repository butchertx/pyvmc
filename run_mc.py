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


def afq120_triangle(uc_x, uc_y):
    phi = 2.0 * np.pi / 3.0
    normu = 1.0
    A = [normu, normu * np.cos(phi), normu * np.cos(2.0*phi)]
    B = [0, normu * np.sin(phi), normu * np.sin(2.0*phi)]
    directors_uc1 = np.array([[A[0], A[1], A[2]],
                              [B[0], B[1], B[2]],
                              [0, 0, 0]])
    stack1 = np.tile(directors_uc1, (1, uc_y))
    directors_uc2 = np.array([[A[2], A[0], A[1]],
                              [B[2], B[0], B[1]],
                              [0, 0, 0]])
    stack2 = np.tile(directors_uc2, (1, uc_y))
    directors_uc3 = np.array([[A[1], A[2], A[0]],
                              [B[1], B[2], B[0]],
                              [0, 0, 0]])
    stack3 = np.tile(directors_uc3, (1, uc_y))
    uc_filled = np.concatenate([stack1, stack2, stack3], axis=1)
    directors_u = np.tile(uc_filled, (1, uc_x))
    return directors_u


def afm120_triangle(uc_x, uc_y):
    phi = 2.0 * np.pi / 3.0
    normu = 1.0 / np.sqrt(2.0)
    A = [normu, normu * np.cos(phi), normu * np.cos(2.0*phi)]
    B = [0, normu * np.sin(phi), normu * np.sin(2.0*phi)]
    directors_uc1 = np.array([[A[0], A[1], A[2]],
                              [B[0], B[1], B[2]],
                              [0, 0, 0]])
    stack1 = np.tile(directors_uc1, (1, uc_y))
    directors_uc2 = np.array([[A[2], A[0], A[1]],
                              [B[2], B[0], B[1]],
                              [0, 0, 0]])
    stack2 = np.tile(directors_uc2, (1, uc_y))
    directors_uc3 = np.array([[A[1], A[2], A[0]],
                              [B[1], B[2], B[0]],
                              [0, 0, 0]])
    stack3 = np.tile(directors_uc3, (1, uc_y))
    uc_filled = np.concatenate([stack1, stack2, stack3], axis=1)
    directors_u = np.tile(uc_filled, (1, uc_x))

    directors_uc = np.array([[0, 0, 0],
                             [0, 0, 0],
                             [normu, normu, normu]])
    stack1 = np.tile(directors_uc, (1, uc_y))
    uc_filled = np.concatenate([stack1, stack1, stack1], axis=1)
    directors_v = np.tile(uc_filled, (1, uc_x))
    return directors_u, directors_v


def create_lattice():
    return lattice.TriangleLattice('triangle', 3*UCY, 3*UCX, unit_cell_mult=1)


def create_wavefunction(lattice_in, state_type='afq3'):
    conf_init = {
        'size': lattice_in.N,
        'S2': 2,
        'num_each': (3*UCX*UCY, 3*UCX*UCY, 3*UCX*UCY)
    }
    jastrow_init = [
        {
            'couples_to': 'sz',
            'strength': 0.0,
            'neighbors': lattice_in.get_neighbor_list(distance_index=j)
        }
        for j in range(2)
    ]
    # jastrow_init = [
    #     {
    #         'couples_to': 'sz',
    #         'strength': -0.2,
    #         'neighbors': lattice_in.get_neighbor_list(distance_index=0)
    #     },
    #     {
    #         'couples_to': 'sz',
    #         'strength': 0.1,
    #         'neighbors': lattice_in.get_neighbor_list(distance_index=1)
    #     }
    # ]
    if state_type == 'afq3':
        directors_u = afq3_triangle(UCX, UCY)
        d_rotate = wavefunction.euler_s1(-3.0 * np.pi / 4.0, np.arccos(np.sqrt(1.0 / 3.0)), 3.0 * np.pi / 4.0, directors_u,
                                         sz_basis=False)
        afq3 = wavefunction.ProductState(conf_init, d_rotate, jastrow_init)
        return afq3
    elif state_type == 'afq120':
        directors_u, directors_v = afm120_triangle(UCX, UCY)
        d_rotate = wavefunction.euler_s1(-3.0 * np.pi / 4.0, np.arccos(np.sqrt(1.0 / 3.0)), 3.0 * np.pi / 4.0,
                                         directors_u,
                                         sz_basis=False)
        afq120 = wavefunction.ProductState(conf_init, d_rotate, jastrow_init)
        return afq120
    elif state_type == 'afm120':
        directors_u, directors_v = afm120_triangle(UCX, UCY)
        directors_d = directors_u + 1j*directors_v
        afm120 = wavefunction.ProductState(conf_init, directors_d, jastrow_init)
        return afm120
    else:
        print('No valid state selected')
        exit()


def create_hamiltonian(lattice_in):
    neighbor_pairs = list(set(map(tuple, map(sorted, lattice_in.get_neighbor_pairs(0))))) #  remove duplicate neighbor pairs
    hermitian_conj_rings = [(sites[0], sites[2], sites[1]) for sites in lattice_in.get_ring_exchange_list()]
    H = local_operator.Hamiltonian()
    # H.add_term('J', 1.0, neighbor_pairs)
    H.add_term('K', 1.0, lattice_in.get_ring_exchange_list(), interaction_type=local_operator.ThreeRingExchange)
    H.add_term('K_prime', 1.0, hermitian_conj_rings, interaction_type=local_operator.ThreeRingExchange)
    return H


if __name__ == '__main__':
    lattice_run = create_lattice()
    wavefunction_run = create_wavefunction(lattice_run, 'afq120')
    ham_run = create_hamiltonian(lattice_run)
    mc = montecarlo.MonteCarlo(wavefunction_run, lattice_run.get_neighbor_pairs(0), measures=1000)
    mc.add_observable(ham_run)
    results, per_site, measurements = mc.run()
    print(per_site)
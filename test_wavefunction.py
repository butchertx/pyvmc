import lattice
import wavefunction
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)


LAT_1 = lattice.Lattice('triangle', 3, 3, unit_cell_mult=1)
NEIGHBOR_LIST_1 = LAT_1.get_neighbor_list(distance_index=0)
CONF_INIT_1 = {
        'size': LAT_1.N,
        'S2': 2,
        'num_each': (3, 3, 3),
        'initial': [1, 1, 1, 0, 0, 0, -1, -1, -1]
    }


def exchange2(site_pair, configuration):
    return [{'site': site_pair[0], 'old_spin': configuration[site_pair[0]], 'new_spin': configuration[site_pair[1]]},
             {'site': site_pair[1], 'old_spin': configuration[site_pair[1]], 'new_spin': configuration[site_pair[0]]}]


def exchange3(site_triple, configuration):
    return [{'site': site_triple[0], 'old_spin': configuration[site_triple[0]], 'new_spin': configuration[site_triple[2]]},
            {'site': site_triple[1], 'old_spin': configuration[site_triple[1]], 'new_spin': configuration[site_triple[0]]},
            {'site': site_triple[2], 'old_spin': configuration[site_triple[2]], 'new_spin': configuration[site_triple[1]]}]


def test_product_state():
    directors_u = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [0, 1, 0, 0, 0, 1, 1, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 1, 0]])
    directors_v = np.zeros((3, 9))
    # print(directors_u)
    # print(directors_v)

    wf_imag = wavefunction.ProductState(CONF_INIT_1, directors_v + 1j * directors_u)

    test_passes = 0
    test_total = 0
    failed = []

    # test 1: rotate the afq3 directors so they have nonzero overlaps on every site with every flavor
    # test will pass if the rotated directors have nonzero entries in sz basis and are mutually orthogonal
    # within a unit cell
    d_rotate = wavefunction.euler_s1(-3.0*np.pi/4.0, np.arccos(np.sqrt(1.0/3.0)), 3.0*np.pi/4.0, directors_u, sz_basis=False)
    print(d_rotate)
    print(np.matmul(np.conj(d_rotate[:, :3].T), d_rotate[:, :3]))
    print(np.abs(np.conj(d_rotate[:, :3])*d_rotate[:, :3]))

    afq3 = wavefunction.ProductState(CONF_INIT_1, d_rotate)
    c = afq3.get_conf()
    print(c)
    print(afq3.site_overlaps)
    hi = 0
    pairlist = LAT_1.get_neighbor_pairs(0)
    for i in range(100):
        site1, site2 = pairlist[np.random.randint(len(pairlist))]
        fliplist = exchange2((site1, site2), c)
        pop = afq3.psi_over_psi(fliplist)
        print(pop)
        if np.real(pop) > 0:
            hi += 1
        afq3.update(fliplist)
        # print(afq3.site_overlaps)

    print(hi)

    return test_passes, test_total, failed


def test_uniform_state():
    wf = wavefunction.UniformState(CONF_INIT_1)
    c = wf.get_conf()
    # print(c)
    test_passes = 0
    test_total = 0
    failed = []

    # test 1: psi_over_psi
    test_total += 1
    if wf.psi_over_psi(exchange2((0, 3), c)):
        test_passes += 1
    else:
        failed.append('psi_over_psi()')

    return test_passes, test_total, failed


def test_jastrow():
    # initialize test cases
    wf = wavefunction.UniformState(CONF_INIT_1)
    c = wf.get_conf()
    jastrow = wavefunction.JastrowFactor(couples_to='sz', strength=1.0, neighbors=NEIGHBOR_LIST_1, configuration=c)
    test_passes = 0
    test_total = 0
    failed = []

    # test 1: Initial exp table = zeros
    test_total += 1
    if all(jastrow.exp_table == np.zeros(9)):
        test_passes += 1
    else:
        failed.append('_exp_table')

    # test 2: Initial greedy eval
    test_total += 1
    init_greedy = jastrow.greedy_eval(c)
    if init_greedy == 1.0:
        test_passes += 1
    else:
        failed.append('Initial greedy_eval()')

    # test 3: Lazy eval pre-specified
    test_total += 1
    fliplist = exchange2((0, 3), configuration=c)
    lazy1 = jastrow.lazy_eval(fliplist)
    if lazy1 == np.exp(-1.0):
        test_passes += 1
    else:
        failed.append('First lazy_eval()')

    # test 4: Update and greedy eval, pre-specified
    test_total += 1
    wf.update(fliplist)
    greedy2 = jastrow.greedy_eval(c)
    if greedy2 == np.exp(-1.0):
        test_passes += 1
    else:
        failed.append('Second greedy_eval()')

    # test 5: Match arbitrary lazy and greedy eval
    test_total += 1
    c = wf.get_conf()
    jastrow = wavefunction.JastrowFactor(couples_to='sz', strength=1.0, neighbors=NEIGHBOR_LIST_1, configuration=c)
    site1 = np.random.randint(len(c))
    site2 = site1
    while site2 == site1:
        site2 = np.random.randint(len(c))
    fliplist = exchange2((site1, site2), c)
    lazy_arb = jastrow.lazy_eval(fliplist)
    greedy_arb = jastrow.greedy_eval(c)
    wf.update(fliplist)
    greedy_arb = jastrow.greedy_eval(c) / greedy_arb
    if np.abs(lazy_arb - greedy_arb) < 1e-10:
        test_passes += 1
    else:
        failed.append('Arbitrary matching greedy_eval() and lazy_eval()')

    # test 6: Update tables
    test_total += 1
    num_stoch_test = 100
    passed_stoch_test = 0
    c = wf.get_conf()
    jastrow_lazy = wavefunction.JastrowFactor(couples_to='sz', strength=1.0, neighbors=NEIGHBOR_LIST_1, configuration=c)
    jastrow_greedy = wavefunction.JastrowFactor(couples_to='sz', strength=1.0, neighbors=NEIGHBOR_LIST_1, configuration=c)
    for i in range(num_stoch_test):
        site1 = np.random.randint(len(c))
        site2 = site1
        while site2 == site1:
            site2 = np.random.randint(len(c))

        fliplist = exchange2((site1, site2), c)
        lazy_arb = jastrow_lazy.lazy_eval(fliplist)
        greedy_arb = jastrow_greedy.greedy_eval(c)
        wf.update(fliplist)
        greedy_arb = jastrow_greedy.greedy_eval(c) / greedy_arb
        jastrow_lazy.update_tables(flip_list=fliplist)
        if np.abs(lazy_arb - greedy_arb) < 1e-12:
            passed_stoch_test += 1

    if passed_stoch_test == num_stoch_test:
        test_passes += 1
    else:
        failed.append('Jastrow update_tables()')

    return test_passes, test_total, failed


def test_jastrow_table():
    # initialize test cases
    wf = wavefunction.UniformState(CONF_INIT_1)
    c = wf.get_conf()
    jastrow1 = wavefunction.JastrowFactor(couples_to='sz', strength=1.0, neighbors=NEIGHBOR_LIST_1, configuration=c)
    jastrow2 = wavefunction.JastrowFactor(couples_to='sz', strength=1.0, neighbors=NEIGHBOR_LIST_1, configuration=c)
    jastrow_tab = wavefunction.JastrowTable([jastrow1, jastrow2])
    test_passes = 0
    test_total = 0
    failed = []

    # test 1: does jastrow ratio from table calculation match square of individuals
    test_total += 1
    num_stoch_test = 100
    passed_stoch_test = 0
    for i in range(num_stoch_test):
        site1 = np.random.randint(len(c))
        site2 = site1
        while site2 == site1:
            site2 = np.random.randint(len(c))

        fliplist = exchange2((site1, site2), c)
        lazy_arb = jastrow1.lazy_eval(fliplist)
        table_arb = jastrow_tab.lazy_eval(fliplist)
        jastrow_tab.update_tables(flip_list=fliplist)

        if np.abs(lazy_arb*lazy_arb - table_arb) < 1e-12:
            passed_stoch_test += 1

    if passed_stoch_test == num_stoch_test:
        test_passes += 1
    else:
        failed.append('JastrowTable update_tables()')

    return test_passes, test_total, failed


print('Running Wavefunction Tests...')


print('Testing Product State...')
passed, total, failed = test_product_state()
print('ProductState Wavefunction Test: Passed ' + str(passed) + ' out of ' + str(total) + ' tests.  Tests Failed: ' + str(failed))

print('Testing Uniform Wavefunction...')
passed, total, failed = test_uniform_state()
print('Uniform Wavefunction Test: Passed ' + str(passed) + ' out of ' + str(total) + ' tests.  Tests Failed: ' + str(failed))

print('Testing Jastrow Factors...')
passed, total, failed = test_jastrow()
print('Jastrow Test: Passed ' + str(passed) + ' out of ' + str(total) + ' tests.  Tests Failed: ' + str(failed))

print('Testing JastrowTable...')
passed, total, failed = test_jastrow_table()
print('JastrowTable Test: Passed ' + str(passed) + ' out of ' + str(total) + ' tests.  Tests Failed: ' + str(failed))
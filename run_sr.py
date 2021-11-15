import lattice
import wavefunction
import local_operator
import montecarlo
import run_mc
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)


UCX = 2
UCY = 2


def default_sr():
    pass


if __name__ == '__main__':
    lattice_run = run_mc.create_lattice()
    wavefunction_run = run_mc.create_wavefunction(lattice_run, 'afq120')
    ham_run = run_mc.create_hamiltonian(lattice_run)
    mc = montecarlo.VariationalMonteCarlo(wavefunction_run, lattice_run.get_neighbor_pairs(0)) # , mc_kwargs={steps_per_measure=10, measures=1000, throwaway=100})
    mc.add_observable(ham_run)
    results, per_site, measurements = mc.run()
    print(per_site)
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
    return montecarlo.StochasticReconfiguration(bins=100, measures_per_bin=100, timestep=0.1)


if __name__ == '__main__':
    lattice_run = run_mc.create_lattice()
    wavefunction_run = run_mc.create_wavefunction(lattice_run, 'afq120')
    ham_run = run_mc.create_hamiltonian(lattice_run)
    mc = montecarlo.VariationalMonteCarlo(wavefunction_run, lattice_run.get_neighbor_pairs(0), sr_object=default_sr())
    mc.add_observable(ham_run)
    results, per_site, measurements = mc.run()
    print(per_site)
    print(mc.SR.params_bins)
    print(mc.SR.energy_bins)
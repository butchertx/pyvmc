'''
Proposing moves based on permutations and/or spin flips
Accepting/Rejecting a move
Holding Measurement Results
Outputting Data, Calculating Averages and Errors
'''
import numpy as np
import wavefunction


def reorder_from_idx(idx, a):
    return a[idx:] + a[:idx]


class MonteCarlo:
    def __init__(self, wf, neighbor_list, steps_per_measure=10, measures=1000, throwaway=100):
        '''
        :param wf: Wavefunction
        :param neighbor_list: dictionary with lists of allowed site permutations.  Should come from Lattice.  For example, list of nn pairs
        :param steps_per_measure: Steps/moves between each MC measurement
        '''
        self.wf = wf
        self.steps_per_measure = steps_per_measure
        self.measures = measures
        self.throwaway = throwaway
        self.neighbor_list = neighbor_list
        self.observables = {}

    def add_observable(self, operator):
        self.observables[operator.name] = operator

    def propose_permutation(self):
        perm = self.neighbor_list[np.random.randint(len(self.neighbor_list))]
        return [{'site': i, 'old_spin': self.wf.configuration[i], 'new_spin': self.wf.configuration[j]} for i, j in zip(perm, reorder_from_idx(1, perm))]

    def propose_flip(self, num_sites):
        pass

    def propose_move(self, move_type='permutation', num_sites=2):
        if move_type == 'permutation':
            return self.propose_permutation()
        elif move_type == 'spin flip':
            return self.propose_flip(num_sites)
        else:
            raise ValueError("Monte Carlo move must be a 'permutation', or a 'spin flip'")

    def step(self):
        flip_list = self.propose_move()
        p_accept = self.wf.psi_over_psi(flip_list)
        if np.abs(p_accept)*np.abs(p_accept) > np.random.uniform():
            self.wf.update(flip_list)

    def run(self):
        measurements = {name: (0.+0.j)*np.ones(self.measures) for name in self.observables}
        for m in range(self.measures):
            if np.mod(m, 1000) == 0:
                print('Measurement ' + str(m) + ' out of ' + str(self.measures))
            for s in range(self.steps_per_measure):
                self.step()

            for name in measurements.keys():
                measurements[name][m] = self.observables[name].local_eval(self.wf)

        averages = {name: np.mean(measurements[name][self.throwaway:]) for name in measurements.keys()}
        per_site = {name: (1./self.wf.configuration.size) * np.mean(measurements[name][self.throwaway:]) for name in measurements.keys()}
        return averages, per_site, measurements


class StochasticReconfiguration:

    def __init__(self, bins=10, measures_per_bin=100, timestep=0.1):
        self.bins = bins
        self.measures_per_bin = measures_per_bin
        self.timestep = timestep
        self.params_bins = []
        self.energy_bins = []

    def save_bin(self, param_vals, energy_avg):
        if len(self.params_bins) == 0:
            self.params_bins = [param_vals]
        else:
            self.params_bins = np.concatenate((self.params_bins, [param_vals]))
        self.energy_bins = np.concatenate((self.energy_bins, [energy_avg]))


class VariationalMonteCarlo(MonteCarlo):
    def __init__(self, wf, neighbor_list, mc_kwargs=None, sr_object=None):
        """
        MonteCarlo class for interfacing with Stochastic Reconfiguration
        :param wf: wavefunction to be passed to the MonteCarlo parent
        :param neighbor_list: list of neighbors to be passed to parent
        :param mc_kwargs: kwargs for MonteCarlo object
        """
        if mc_kwargs is not None:
            MonteCarlo.__init__(self, wf, neighbor_list, **mc_kwargs)
        else:
            MonteCarlo.__init__(self, wf, neighbor_list)

        if sr_object is None:
            self.SR = StochasticReconfiguration()
        else:
            self.SR = sr_object

    def run_sr_iteration(self):
        measurements = {name: np.zeros((self.SR.measures_per_bin,), dtype=np.complex64) for name in self.observables}
        ok_measurements = np.zeros((self.SR.measures_per_bin, self.wf.calculate_log_derivative().size), dtype=np.complex64)
        for m in range(self.SR.measures_per_bin):
            for s in range(self.steps_per_measure):
                self.step()

            for name in measurements.keys():
                measurements[name][m] = self.observables[name].local_eval(self.wf)

            ok_measurements[m, :] = self.wf.calculate_log_derivative()

        try:
            local_energies = np.real(measurements['Hamiltonian'])
        except KeyError:
            print('Hamiltonian must be defined and named in order to use the Variational MonteCarlo!')
            raise NameError('Hamiltonian')

        ok_mean = np.mean(ok_measurements, axis=0)
        e_mean = np.mean(local_energies)
        sjk = np.tensordot(np.matrix(ok_measurements).H, ok_measurements, axes=1) / self.SR.measures_per_bin
        sjk0 = np.outer(sjk[0, :], sjk[0, :])
        sjk = sjk - sjk0
        fk = ok_mean * e_mean - np.tensordot(np.matrix(ok_measurements).H, local_energies, axes=1) / self.SR.measures_per_bin
        del_alpha = np.linalg.solve(sjk[1:, 1:], fk[1:])
        old_params = self.wf.get_params()
        new_params = old_params + self.SR.timestep * np.real(del_alpha)
        self.SR.save_bin(old_params, e_mean / self.wf.configuration.size)
        self.wf.update_parameters(new_params)
        print('Update Parameters')
        print('Old Parameters:', old_params)
        print('New Parameters:', new_params)
        return measurements

    def run(self):
        measurements = {name: (0. + 0.j) * np.ones(self.SR.bins * self.SR.measures_per_bin) for name in self.observables}
        for sri in range(self.SR.bins):
            print('SR Iteration ' + str(sri+1) + ' out of ' + str(self.SR.bins))
            m_iteration = self.run_sr_iteration()
            print('SR bin measurements:')
            for name in self.observables:
                measurements[name][sri*len(m_iteration[name]):(sri+1)*len(m_iteration[name])] = m_iteration[name]
                print(name + ' = ' + str(np.mean(m_iteration[name])))

        averages = {name: np.mean(measurements[name][self.throwaway:]) for name in measurements.keys()}
        per_site = {name: (1. / self.wf.configuration.size) * np.mean(measurements[name][self.throwaway:]) for name in
                    measurements.keys()}
        return averages, per_site, measurements

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
    def __init__(self, wf, neighbor_list, steps_per_measure=10):
        '''
        :param neighbor_list: dictionary with lists of allowed site permutations.  Should come from Lattice.  For example, list of nn pairs
        :param steps_per_measure: Steps/moves between each MC measurement
        '''
        self.wf = wf
        self.steps_per_measure = steps_per_measure
        self.neighbor_list = neighbor_list

    def propose_permutation(self, num_sites):
        perm = self.neighbor_list[num_sites][np.random.randint(self.wf.conf.size)]
        print(perm)
        while len(set([self.wf.conf[i] for i in perm])) == 1:
            perm = self.neighbor_list[num_sites][np.random.randint(self.wf.conf.size)]
            print(perm)

        return [{'site': i, 'old_spin': self.wf.conf[i], 'new_spin': self.wf.conf[j]} for i, j in zip(perm, reorder_from_idx(1, perm))]

    def propose_flip(self, num_sites):
        pass

    def propose_move(self, move_type='permutation', num_sites=2):
        if move_type == 'permutation':
            return self.propose_permutation(num_sites)
        elif move_type == 'spin flip':
            return self.propose_flip(num_sites)
        else:
            raise ValueError("Monte Carlo move must be a 'permutation', or a 'spin flip'")

    def step(self):
        flip_list = self.propose_move()
        p_accept = self.wf.psi_over_psi(flip_list)
        if p_accept > np.random.uniform():
            self.wf.update(flip_list)
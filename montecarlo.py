'''
Random configurations in the Sz-basis
Proposing moves based on permutations and/or spin flips
Accepting/Rejecting a move
Holding Measurement Results
Outputting Data, Calculating Averages and Errors
'''
import numpy as np


def is_odd(num):
    return num & 0x1


class Configuration:
    def __init__(self, size, S2, num_each, initial=None):
        # size = number of elements
        # S2 = 2S, or 1 for spin-1/2, 2 for spin-1, etc.
        # num_each = tuple(n(-S),n(-S+1)...n(S-1),n(S)).  Default equal numbers
        # initial = initial configuration
        assert (S2 + 1 == len(num_each))
        assert (sum(num_each) == size)
        self.size = size
        self.S2 = S2
        self.num_each = num_each
        self.half_integer = is_odd(self.S2)
        self.conf = self.random_conf()

    def random_conf(self):
        idx = np.random.permutation(np.arange(self.size))
        sections = np.cumsum(self.num_each)
        partition = np.split(idx, sections)[:-1]
        configuration = np.array([0]*self.size)
        if self.half_integer:
            spin_val = -self.S2
        else:
            spin_val = -np.rint(self.S2/2)

        for p in partition:
            configuration[p] = spin_val
            spin_val += 1

        return configuration


class MonteCarlo:
    def __init__(self, steps_per_measure=10):
        self.steps_per_measure = steps_per_measure


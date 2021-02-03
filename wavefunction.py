'''
Initialize and update configurations (basis states) in the Sz-basis
Calculate ratios of overlaps based on spin flips
'''
import numpy as np


def is_odd(num):
    return num & 0x1


class Configuration:

    _size = 2
    _S2 = 1
    _num_each = 1
    _half_integer = True
    _conf = np.array([1, -1])

    def __init__(self, size, S2, num_each, initial=None):
        # size = number of elements
        # S2 = 2S, or 1 for spin-1/2, 2 for spin-1, etc.
        # num_each = tuple(n(-S),n(-S+1)...n(S-1),n(S)).  Default equal numbers
        # initial = initial configuration
        assert (S2 + 1 == len(num_each))
        assert (sum(num_each) == size)
        self._size = size
        self._S2 = S2
        self._num_each = num_each
        self._half_integer = is_odd(self._S2)
        if initial is None:
            self._conf = self.random_conf()
        else:
            self._conf = initial
            # TODO: check initialization to ensure consistency with other inputs

    def __str__(self):
        return str(self._conf)

    def __getitem__(self, key):
        return self._conf[key]

    def random_conf(self):
        idx = np.random.permutation(np.arange(self._size))
        sections = np.cumsum(self._num_each)
        partition = np.split(idx, sections)[:-1]
        configuration = np.array([0] * self._size)
        if self._half_integer:
            spin_val = -self._S2
        else:
            spin_val = -np.rint(self._S2 / 2)

        for p in partition:
            configuration[p] = spin_val
            spin_val += 1

        return configuration

    def update(self, flip_list):
        for item in flip_list:
            assert (self._conf[item['site']] == item['old_spin'])
            self._conf[item['site']] = item['new_spin']

    def get_conf(self):
        return self._conf

    def get_sz(self, site):
        return 0.5 * self._S2 * self._conf[site]


class Wavefunction(object):

    _configuration = Configuration(2, 1, (1, 1))

    def __init__(self, conf_init):
        """
        :param conf_init: kwargs for initializing the configuration
        """
        self._configuration = Configuration(**conf_init)

    def psi_over_psi(self, flip_list):
        raise NotImplementedError('psi_over_psi must be defined for your Wavefunction!')

    def update(self, flip_list):
        raise NotImplementedError('update must be defined for your Wavefunction!')

    def get_conf(self):
        return self._configuration.get_conf()


class ProductState(Wavefunction):

    def __init__(self, conf_init, directors):
        """
        Site-factorized state of directors
        :param conf_init: kwargs for initializing the configuration
        :param directors: List of directors by site.  A director is a complex vector with 2S+1 elements
        """
        super(ProductState, self).__init__(self, conf_init)

    def psi_over_psi(self, flip_list):
        pass

    def update(self, flip_list):
        pass


class UniformState(Wavefunction):
    def __init__(self, conf_init):
        """
        Uniform wavefunction.  Every state is equally likely (psi = 1, const.)
        :param conf_init: kwargs for initializing the configuration
        """
        Wavefunction.__init__(self, conf_init)

    def psi_over_psi(self, flip_list):
        return 1.0

    def update(self, flip_list):
        self._configuration.update(flip_list)


class JastrowFactor:

    _couples_to = 'sz'
    _strength = 0.0
    _neighbor_table = np.array([0.0])
    _exp_table = np.array([0.0])

    def __init__(self, couples_to, strength, neighbors, configuration):
        """
        Jastrow table with associated coupling.  J = exp(1/2 sum_ij v O_i O_j)
        :param couples_to: diagonal operator (sz, sz2, etc.)
        :param strength: coupling strength v
        :param neighbors: list of sites and their neighbors associated with this factor
        :param configuration: initial configuration for setting the table
        """
        if couples_to != 'sz':
            raise NotImplementedError('Jastrow Factor must couple to Sz!')

        self._couples_to = couples_to
        self._strength = strength
        self._neighbor_table = neighbors
        self.initialize_table(configuration)

    def initialize_table(self, configuration):
        """
        table of site-sums.  exp_table[i] = sum(v * sum_j O_j)
        J = exp(exp_table[i] dot O(conf[i]))
        """
        self._exp_table = np.array([np.sum(self._strength * configuration[j] for j in neighborlist) for neighborlist in self._neighbor_table])

    def greedy_eval(self, configuration):
        self.initialize_table(configuration)
        return np.exp(0.5*np.dot(self._exp_table, configuration))

    def lazy_eval(self, flip_list):
        flip_sum = 0.0
        neighbor_sum = 0.0
        for flip in flip_list:
            del_s = flip['new_spin'] - flip['old_spin']
            flip_sum += np.sum([self._strength * del_s * flip2['new_spin'] for flip2 in flip_list if
                                flip2['site'] in self._neighbor_table[flip['site']]])
            neighbor_sum += del_s * self._exp_table[flip['site']]

        return np.exp(flip_sum + neighbor_sum)

    def update_tables(self, flip_list):
        flip_sites = [flip['site'] for flip in flip_list]
        del_S = [flip['new_spin'] - flip['old_spin'] for flip in flip_list]
        update_list = np.zeros(len(self._neighbor_table))
        for flipsite, dels in zip(flip_sites, del_S):
            update_list[flipsite] = self._strength * dels

        for idx in range(len(self._neighbor_table)):
            self._exp_table[idx] += np.sum(update_list[self._neighbor_table[idx]])


class JastrowTable:

    _jastrows = []

    def __init__(self, jastrow_list):
        self._jastrows = jastrow_list

    def greedy_eval(self, configuration):
        return np.prod([jast.greedy_eval(configuration) for jast in self._jastrows])

    def lazy_eval(self, flip_list):
        return np.prod([jast.lazy_eval(flip_list) for jast in self._jastrows])

    def update_tables(self, flip_list):
        for jast in self._jastrows:
            jast.update_tables(flip_list)
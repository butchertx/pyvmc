'''
Initialize and update configurations (basis states) in the Sz-basis
Calculate ratios of overlaps based on spin flips
'''
import numpy as np


SQRT_2 = np.sqrt(2)
SQRT_HALF = np.sqrt(0.5)


def is_odd(num):
    return num & 0x1


def sz_director_basis(d):
    if d.shape[0] == 3:
        return np.array([SQRT_HALF*(-1j*d[0, :] + d[1, :]),
                         1j*d[2, :],
                         SQRT_HALF*(1j*d[0, :] + d[1, :])])
    else:
        raise NotImplementedError('Only allowing S=1 directors for now')


def quad_director_basis(d):
    if d.shape[0] == 3:
        return np.array([SQRT_HALF * 1j * (d[0, :] - d[2, :]),
                         SQRT_HALF * (d[0, :] + d[2, :]),
                         -1j * d[1, :]])
    else:
        raise NotImplementedError('Only allowing S=1 directors for now')


def euler_s1(alpha, beta, gamma, d_list, sz_basis=False):
    '''
    perform an Euler rotation on the directors d_list.  Defined for example in section 3.5 of Sakurai
    (pg 198, eqs 3.5.50 and 3.5.57)
    :param alpha: Euler angle alpha (second rotation around Jz)
    :param beta: Euler angle beta (rotation around Jy)
    :param gamma: Euler angle gamma (first rotation around Jz)
    :param d_list: List of directors to rotate
    :param sz_basis: By default, directors are in quadrupolar basis and need to be transformed to sz-basis.
    This parameter is True if d_list is in Sz basis
    :return: rotated d_list
    '''
    if not sz_basis:
        d_list = sz_director_basis(d_list)

    phase_m = np.array([[1., 0., -1.],
                        [1., 0., -1.],
                        [1., 0., -1.]])
    phases = np.exp(-1j*gamma*phase_m - 1j*alpha*phase_m.T)
    sinb = np.sin(beta)
    cosb = np.cos(beta)
    d_mat = np.array([[0.5*(1 + cosb), -SQRT_HALF*sinb, 0.5*(1 - cosb)],
                      [SQRT_HALF*sinb, cosb, -SQRT_HALF*sinb],
                      [0.5*(1 - cosb), SQRT_HALF*sinb, 0.5*(1 + cosb)]])
    euler_mat = phases * d_mat
    # print(np.matmul(np.conj(euler_mat.T), euler_mat))
    d_transform = np.matmul(euler_mat, d_list)
    if not sz_basis:
        d_transform = quad_director_basis(d_transform)

    return d_transform


class Configuration:

    size = 2
    S2 = 1
    num_each = 1
    half_integer = True
    conf = np.array([1, -1])

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
        if initial is None:
            self.conf = self.random_conf()
        else:
            self.conf = initial.copy()
            # TODO: check initialization to ensure consistency with other inputs

    def __str__(self):
        return str(self.conf)

    def __getitem__(self, key):
        return self.conf[key]

    def random_conf(self):
        idx = np.random.permutation(np.arange(self.size))
        sections = np.cumsum(self.num_each)
        partition = np.split(idx, sections)[:-1]
        configuration = np.array([0] * self.size)
        if self.half_integer:
            spin_val = -self.S2
        else:
            spin_val = -np.rint(self.S2 / 2)

        for p in partition:
            configuration[p] = spin_val
            spin_val += 1

        return configuration

    def update(self, flip_list):
        for item in flip_list:
            assert (self.conf[item['site']] == item['old_spin'])
            self.conf[item['site']] = item['new_spin']

    def get_conf(self):
        return self.conf

    def get_sz(self, site):
        return 0.5 * self.S2 * self.conf[site]

    def sz_conf_idx(self, m):
        if self.half_integer:
            return -m + self.S2
        else:
            return -m + int(0.5*self.S2)

    def get_sz_idx(self, site):
        return self.sz_conf_idx(self.conf[site])


class JastrowFactor:

    couples_to = 'sz'
    strength = 0.0
    neighbor_table = np.array([0.0])
    exp_table = np.array([0.0])
    conf_sum = 0.0

    def __init__(self, couples_to, strength, neighbors, configuration):
        """
        Jastrow table with associated coupling.  J = exp(1/2 sum_ij v O_i O_j)
        :param couples_to: diagonal operator (sz, sz2, etc.)
        :param strength: coupling strength v
        :param neighbors: list of sites and their neighbors associated with this factor
        :param configuration: (ndarray) initial configuration for setting the table
        """
        if couples_to != 'sz':
            raise NotImplementedError('Jastrow Factor must couple to Sz!')

        self.couples_to = couples_to
        self.strength = strength
        self.neighbor_table = neighbors
        self.initialize_table(configuration)

    def initialize_table(self, configuration):
        """
        table of site-sums.  exp_table[i] = sum(sum_j O_j)
        J = exp(v * exp_table[i] dot O(conf[i]))
        """
        self.exp_table = np.array([np.sum(configuration[j] for j in neighborlist) for neighborlist in self.neighbor_table])
        self.conf_sum = 0.5 * np.dot(self.exp_table, configuration)

    def greedy_eval(self, configuration):
        self.initialize_table(configuration)
        return np.exp(0.5*self.strength*np.dot(self.exp_table, configuration))

    def lazy_eval(self, flip_list):
        flip_sum = 0.0
        neighbor_sum = 0.0
        for flip in flip_list:
            del_s = flip['new_spin'] - flip['old_spin']
            flip_sum += np.sum([self.strength * del_s * flip2['new_spin'] for flip2 in flip_list if
                                flip2['site'] in self.neighbor_table[flip['site']]])
            neighbor_sum += del_s * self.strength * self.exp_table[flip['site']]

        return np.exp(flip_sum + neighbor_sum)

    def update_tables(self, flip_list):
        flip_sites = [flip['site'] for flip in flip_list]
        del_S = [flip['new_spin'] - flip['old_spin'] for flip in flip_list]
        update_list = np.zeros(len(self.neighbor_table))
        for flipsite, dels in zip(flip_sites, del_S):
            update_list[flipsite] = dels

        for idx in range(len(self.neighbor_table)):
            self.exp_table[idx] += np.sum(update_list[self.neighbor_table[idx]])

        flip_sum = 0.0
        neighbor_sum = 0.0
        for flip in flip_list:
            del_s = flip['new_spin'] - flip['old_spin']
            flip_sum += np.sum([del_s * flip2['new_spin'] for flip2 in flip_list if
                                flip2['site'] in self.neighbor_table[flip['site']]])
            neighbor_sum += del_s * self.exp_table[flip['site']]

        self.conf_sum = flip_sum + neighbor_sum

    def calculate_log_derivative(self):
        """
        :return: the logarithmic derivative of this Jastrow factor
        """
        return self.conf_sum

    def get_param(self):
        return self.strength

    def set_param(self, new_param):
        self.strength = new_param


class JastrowTable:

    jastrows = []

    def __init__(self, jastrow_list=None, jastrow_kwargs_list=None):
        if jastrow_list is not None:
            self.jastrows = jastrow_list
        elif jastrow_kwargs_list is not None:
            self.jastrows = [JastrowFactor(**kwargs) for kwargs in jastrow_kwargs_list]
        else:
            raise RuntimeError('Must enter a valid parameter to JastrowTable()')

    def greedy_eval(self, configuration):
        return np.prod([jast.greedy_eval(configuration) for jast in self.jastrows])

    def lazy_eval(self, flip_list):
        return np.prod([jast.lazy_eval(flip_list) for jast in self.jastrows])

    def update_tables(self, flip_list):
        for jast in self.jastrows:
            jast.update_tables(flip_list)

    def calculate_log_derivative(self):
        """
        :return: logarithmic derivative of the Jastrow table (array containing one value for each variational param)
        """
        return np.array([jast.calculate_log_derivative() for jast in self.jastrows])

    def get_params(self):
        return np.array([jast.get_param() for jast in self.jastrows])

    def set_params(self, new_params):
        for param, jast in zip(new_params, self.jastrows):
            jast.set_param(param)


class Wavefunction(object):

    configuration = Configuration(2, 1, (1, 1))

    def __init__(self, conf_init):
        """
        :param conf_init: kwargs for initializing the configuration
        """
        self.configuration = Configuration(**conf_init)

    def psi_over_psi(self, flip_list):
        raise NotImplementedError('psi_over_psi must be defined for your Wavefunction!')

    def update(self, flip_list):
        raise NotImplementedError('update must be defined for your Wavefunction!')

    def get_conf(self):
        return self.configuration.get_conf()

    def calculate_log_derivative(self):
        """
        For a variational wavefunction, calculate the logarithmic derivative at the current configuration
        with respect to the variational parameters
        :return: numpy array with the derivative of parameter k at index k.  Return 1 at index 0 (by definition
        of the SR algorithm, the 0th variational parameter corresponds to the identity operator)
        """
        return np.ones(1)

    def update_parameters(self):
        raise NotImplementedError('update_parameters must be defined for your Wavefunction to be variational!')


class ProductState(Wavefunction):

    jastrow_table = None

    def __init__(self, conf_init, directors, jastrow_init=None):
        """
        Site-factorized state of directors
        :param conf_init: kwargs for initializing the configuration
        :param directors: numpy array of directors by site.  A director is a complex vector with 2S+1 elements
        :param jastrow_init: kwargs for initalizing the jastrow factors
        """
        Wavefunction.__init__(self, conf_init)
        assert(self.configuration.size == directors.shape[1])
        self.directors = directors

        # normalize
        norms = np.sum(self.directors * np.conj(self.directors), 0)
        self.directors = self.directors / np.sqrt(norms)
        self.directors_sz = sz_director_basis(self.directors)
        self.site_overlaps = np.array([self.directors_sz[self.configuration.get_sz_idx(site), site]
                                       for site in range(self.configuration.size)])

        # initalize the jastrow factors
        if jastrow_init is not None:
            for factor in jastrow_init:
                if 'configuration' not in factor.keys():
                    factor['configuration'] = self.configuration.get_conf()
            self.jastrow_table = JastrowTable(jastrow_kwargs_list=jastrow_init)

    def psi_over_psi(self, flip_list):
        old_prod = np.prod([self.site_overlaps[flip['site']] for flip in flip_list])
        new_prod = np.prod([self.directors_sz[self.configuration.sz_conf_idx(flip['new_spin']), flip['site']]
                            for flip in flip_list])
        if self.jastrow_table is not None:
            jastrow_ratio = self.jastrow_table.lazy_eval(flip_list)
        else:
            jastrow_ratio = 1.0

        return jastrow_ratio * np.divide(new_prod, old_prod)

    def update(self, flip_list):
        self.configuration.update(flip_list)
        if self.jastrow_table is not None:
            self.jastrow_table.update_tables(flip_list)

        for flip in flip_list:
            self.site_overlaps[flip['site']] = self.directors_sz[self.configuration.get_sz_idx(flip['site']), flip['site']]

    def calculate_log_derivative(self):
        """
        INSERT CODE HERE.  This should call the function of the same name from the Jastrow table
        :return: numpy array with the derivative of parameter k at index k.  Return 1 at index 0 (by definition
        of the SR algorithm, the 0th variational parameter corresponds to the identity operator)
        """
        return np.concatenate(([1.0], self.jastrow_table.calculate_log_derivative()))

    def get_params(self):
        return self.jastrow_table.get_params()

    def update_parameters(self, new_params):
        self.jastrow_table.set_params(new_params)


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
        self.configuration.update(flip_list)



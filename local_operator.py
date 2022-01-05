import numpy as np
import wavefunction


def exchange2(site_pair, configuration):
    return [{'site': site_pair[0], 'old_spin': configuration[site_pair[0]], 'new_spin': configuration[site_pair[1]]},
             {'site': site_pair[1], 'old_spin': configuration[site_pair[1]], 'new_spin': configuration[site_pair[0]]}]


def s_plus_minus(site_pair, configuration):
    return [{'site': site_pair[0], 'old_spin': configuration[site_pair[0]], 'new_spin': configuration[site_pair[0]]-1},
             {'site': site_pair[1], 'old_spin': configuration[site_pair[1]], 'new_spin': configuration[site_pair[1]]+1}]


def list_spm_smp(site_pair, configuration, spinmax):
    fliplist_list = []
    if configuration[site_pair[0]] > -spinmax + 0.5 and configuration[site_pair[1]] < spinmax - 0.5:
        fliplist_list.append(s_plus_minus(site_pair, configuration))
    if configuration[site_pair[1]] > -spinmax + 0.5 and configuration[site_pair[0]] < spinmax - 0.5:
        fliplist_list.append(s_plus_minus((site_pair[1], site_pair[0]), configuration))
    return fliplist_list



def exchange3(site_triple, configuration):
    return [{'site': site_triple[0], 'old_spin': configuration[site_triple[0]], 'new_spin': configuration[site_triple[2]]},
            {'site': site_triple[1], 'old_spin': configuration[site_triple[1]], 'new_spin': configuration[site_triple[0]]},
            {'site': site_triple[2], 'old_spin': configuration[site_triple[2]], 'new_spin': configuration[site_triple[1]]}]


class LocalOperator:

    def __init__(self, site_list):
        '''
        Create a local operator acting on a collection of SU(2) objects in the computational basis
        :param site_list: list of sites acted on by operator
        '''
        self.site_list = site_list

    def diag(self, configuration):
        return 0.0

    def off_diag(self, configuration):
        '''
        compute the offdiagonal terms
        :param configuration: Configuration object
        :return: coefficient and list of Fliplist objects
        '''
        pass


class HeisenbergExchange(LocalOperator):

    def __init__(self, site_list, spinval=1):
        """
        Heisenberg Exchange: S_i cdot S_j = S_i^z S_j^z + 0.5(S_i^+S_j^- + S_i^-S_j^+)
        """
        assert(len(site_list) == 2)
        LocalOperator.__init__(self, site_list)
        self.spinval = spinval

    def diag(self, configuration):
        return self.spinval * self.spinval * configuration[self.site_list[0]] * configuration[self.site_list[1]]

    def off_diag(self, configuration):
        # TODO: update for arbitrary spin value (this is correct for S=1)
        return list_spm_smp(self.site_list, configuration, 1), 1.0


class BilinearExchange(LocalOperator):

    def __init__(self, site_list):
        """
        Bilinear Exchange: exchange the spins on site i and site j
        """
        assert(len(site_list) == 2)
        LocalOperator.__init__(self, site_list)

    def off_diag(self, configuration):
        return [exchange2((self.site_list[0], self.site_list[1]), configuration)], 1.0


class ThreeRingExchange(LocalOperator):

    def __init__(self, site_list):
        """
        ThreeRingExchange Exchange: cyclically permute the spins on sites i,j,k (i->j->k->i)
        """
        assert(len(site_list) == 3)
        LocalOperator.__init__(self, site_list)

    def off_diag(self, configuration):
        return [exchange3((self.site_list[0], self.site_list[1], self.site_list[2]), configuration)], 1.0


class Hamiltonian:

    def __init__(self):
        self.term_dict = {}
        self.coupling_dict = {}
        self.name = 'Hamiltonian'

    def add_term(self, name, coupling, neighbor_tuples, interaction_type=BilinearExchange):
        if name in self.term_dict.keys():
            raise RuntimeWarning('Tried to add ' + name + ' terms to Hamiltonian multiple times!')

        self.term_dict[name] = [interaction_type(t) for t in neighbor_tuples]
        self.coupling_dict[name] = coupling

    def local_eval(self, wf):
        result = 0.0
        config = wf.get_conf()
        for term in self.term_dict.keys():
            diag_terms = np.sum([op.diag(config) for op in self.term_dict[term]])
            reslist = [op.off_diag(config) for op in self.term_dict[term]]
            off_diag_terms = 0.0
            # first iterate over terms
            for flips, coeff in reslist:
                # term gives a list of flips and a coeff, get the sum of the flips
                off_diag_terms += np.sum([wf.psi_over_psi(flip_list) * coeff for flip_list in flips])

            result += self.coupling_dict[term] * (diag_terms + off_diag_terms)

        return result

import random as rn
import numpy as np
import tools as tools_modified
from deap import base, creator, tools
from train import auto_encoder


def rand_bin():
    return rn.randint(0, 1)


def sel_rank_roulette(individuals, k=2):
    """Rank-based Roulette selection"""
    fits = []
    for ind in individuals:
        fits.append(1 / (ind.rank + 1))
    fits_sum = sum(fits)
    fits = [fits[i] / fits_sum for i in range(len(fits))]
    selection = set([])
    while len(selection) < k:
        prob = np.random.random()
        sum_ = 0
        rank = -1
        while sum_ < prob:
            rank += 1
            sum_ += fits[rank]
        selection = selection | {int(rank)}
    return [individuals[i] for i in selection]


class CoEA:
    def __init__(self, pop_size_bits, n_layer_species, layer_weights, net_weights, iters, net_pop_size, data):
        # global random seeds
        # (not to be confused with NN weight initialization random seed)
        np.random.seed(0)
        rn.seed(0)

        self.pop_size_bits = pop_size_bits  # Size of layer population needs to be a power of 2
        self.n_layer_species = n_layer_species  # Number of hidden layers fixed to 4
        self.layer_weights = layer_weights
        self.net_weights = net_weights
        self.iters = iters  # AE training iterations
        self.net_pop_size = net_pop_size
        self.layer_pop_size = 2 ** self.pop_size_bits
        self.data = data

        # number of bits allocated for each layer chromosome gene
        self.layer_gene_bits = {'L2': 8,
                                'SP': 8,
                                'SR': 8,
                                'act': 2,
                                'init': 2,
                                'n_neuron': 3}

        # number of bits allocated for each network chromosome gene
        self.net_gene_bits = {'batch': 3,
                              'optim': 2,
                              'learn_rate': 8,
                              'decay': 8,
                              'mom': 5,
                              'rand_seed': 6,
                              'h1': self.pop_size_bits,
                              'h2': self.pop_size_bits,
                              'h3': self.pop_size_bits,
                              'h4': self.pop_size_bits}  # layer genes always at the end of the chromosome

        # range of values for each layer chromosome gene
        self.layer_gene_range = [{'L2': np.logspace(-8, 0, 2 ** self.layer_gene_bits['L2'], dtype='float32'),
                                  'SP': np.linspace(0.001, 0.1, 2 ** self.layer_gene_bits['SP'], dtype='float32'),
                                  'SR': np.logspace(-3, 1, 2 ** self.layer_gene_bits['SR'], dtype='float32'),
                                  'act': ['sigmoid', 'tanh', 'softsign', 'selu'],
                                  'init': ['glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
                                  'n_neuron': np.arange(3, 11)}]

        for i in range(n_layer_species - 1):
            dic = self.layer_gene_range[0].copy()
            self.layer_gene_range.append(dic)

        self.layer_gene_range[1]['n_neuron'] = np.arange(14, 29, 2)
        self.layer_gene_range[2]['n_neuron'] = np.arange(20, 91, 10)
        self.layer_gene_range[3]['n_neuron'] = np.arange(50, 121, 10)

        # range of values for each network chromosome gene
        self.net_gene_range = {'batch': [2 ** i for i in range(8)],
                               'optim': ['adam', 'nadam', 'rmsprop', 'adadelta'],
                               'learn_rate': np.logspace(-8, 1, 2 ** self.net_gene_bits['learn_rate'], dtype='float32'),
                               'decay': np.logspace(-8, 0, 2 ** self.net_gene_bits['decay'], dtype='float32'),
                               'mom': np.logspace(-0.05, -1e-5, 2 ** self.net_gene_bits['mom'], dtype='float32'),
                               'rand_seed': np.arange(2 ** self.net_gene_bits['rand_seed']),
                               'h1': [i for i in range(2 ** self.net_gene_bits['h1'])],
                               'h2': [i for i in range(2 ** self.net_gene_bits['h2'])],
                               'h3': [i for i in range(2 ** self.net_gene_bits['h3'])],
                               'h4': [i for i in range(2 ** self.net_gene_bits['h4'])]}

        # In this version of Python, dictionary is ordered
        self.s_idx_layer = []  # start indexes for layer chromosome genes
        self.e_idx_layer = []  # end indexes for layer chromosome genes
        s = 0
        for i in self.layer_gene_bits:
            self.s_idx_layer.append(s)
            self.e_idx_layer.append(s + self.layer_gene_bits[i])
            s = s + self.layer_gene_bits[i]

        self.s_idx_net = []  # start indexes for network chromosome genes
        self.e_idx_net = []  # end indexes for network chromosome genes
        s = 0
        for i in self.net_gene_bits:
            self.s_idx_net.append(s)
            self.e_idx_net.append(s + self.net_gene_bits[i])
            s = s + self.net_gene_bits[i]

        # chromosome lengths
        self.layer_ind_size = sum(self.layer_gene_bits.values())
        self.net_ind_size = sum(self.net_gene_bits.values())

        # position of the first layer gene in net ind
        self.pos = len(self.net_gene_bits) - n_layer_species

        self.toolbox = base.Toolbox()

        # DEAP library classes definition
        creator.create('FitnessLayer', base.Fitness, weights=self.layer_weights)
        creator.create('FitnessNet', base.Fitness, weights=self.net_weights)
        creator.create('LayerIndividual', list, fitness=creator.FitnessLayer)
        creator.create('NetIndividual', list, fitness=creator.FitnessNet)

        self.toolbox.register('rand_bin', rand_bin)
        self.toolbox.register('layerIndividual', tools.initRepeat, creator.LayerIndividual,
                              self.toolbox.rand_bin, n=self.layer_ind_size)
        self.toolbox.register('netIndividual', tools.initRepeat, creator.NetIndividual,
                              self.toolbox.rand_bin, n=self.net_ind_size)
        self.toolbox.register('layerSpecies', tools.initRepeat, list, self.toolbox.layerIndividual,
                              n=self.layer_pop_size)
        self.toolbox.register('netPopulation', tools.initRepeat, list, self.toolbox.netIndividual,
                              n=self.net_pop_size)
        self.toolbox.register('evaluateNet', self.eval_fitness)
        self.toolbox.register('mate', self.cx_net_layers)
        self.toolbox.register('mutateNetStructure', self.mut_structure)
        self.toolbox.register('mutateNetParameters', self.mut_parameters)
        self.toolbox.register('mutateLayerParameters', self.mut_layer_ind)
        self.toolbox.register("selectNSGA2fronts", tools_modified.selNSGA2, nd='standard')
        self.toolbox.register("selectNSGA2", tools.selNSGA2, nd='standard')
        self.toolbox.register("selectRoulette", sel_rank_roulette)

        # initialize layer population
        self.layer_population = [self.toolbox.layerSpecies() for i in range(n_layer_species)]
        # assign idx for each layer individual
        for layer_species in self.layer_population:
            for i, layer_ind in enumerate(layer_species):
                layer_ind.index = i
        # initialize net population
        self.net_population = self.toolbox.netPopulation()
        # All initial layers should participate in networks
        for i in range(self.layer_pop_size):
            bin_layer_idx = np.binary_repr(i, pop_size_bits)
            for j in range(n_layer_species):
                sidx = self.s_idx_net[j + self.pos]
                eidx = self.e_idx_net[j + self.pos]
                self.net_population[i][sidx:eidx] = list(np.int_(list(bin_layer_idx)))
        for ind in self.net_population:
            ind.age = 0

    def get_net_params(self, net_ind):
        """Returns a dictionary of the parameters corresponding to a network chromosome"""
        net_ind_str = str(net_ind).strip('[]').replace(' ', '').replace(',', '').replace('\n', '')
        net_ind_decimal = []
        for i in range(len(self.net_gene_bits.keys())):
            net_ind_decimal.append(int(net_ind_str[self.s_idx_net[i]:self.e_idx_net[i]], 2))
        net_params = {}
        for i, key in enumerate(self.net_gene_range):
            net_params[key] = self.net_gene_range[key][net_ind_decimal[i]]
        return net_params

    def get_layer_params(self, layer_ind, species_idx):
        """Returns a dictionary of the parameters corresponding to a layer chromosome"""
        layer_ind_str = str(layer_ind).strip('[]').replace(' ', '').replace(',', '').replace('\n', '')
        layer_ind_decimal = []
        for i in range(len(self.layer_gene_bits.keys())):
            layer_ind_decimal.append(int(layer_ind_str[self.s_idx_layer[i]:self.e_idx_layer[i]], 2))
        layer_params = {}
        for i, key in enumerate(self.layer_gene_range[species_idx]):
            layer_params[key] = self.layer_gene_range[species_idx][key][layer_ind_decimal[i]]
        return layer_params

    def eval_fitness(self, net_ind):
        """Calculates the fitness of a network individual
        Input: network individual
        Returns: a tuple of fitness values
        """
        net_params = self.get_net_params(net_ind)
        layer_params_list = []
        for i in range(self.n_layer_species):
            layer_species = self.layer_population[i].copy()
            layer_species.sort(key=lambda x: x.index)
            layer_ind = layer_species[net_params['h' + str(i + 1)]]
            layer_params = self.get_layer_params(layer_ind, i)
            layer_params_list.append(layer_params)
        net_ind.net_params = net_params
        net_ind.layer_params = layer_params_list
        return auto_encoder(net_params, layer_params_list, self.data,
                            n_layers=self.n_layer_species, iters=self.iters)

    def layers_credit_assignment(self, net_pop):
        """Assigns credits to layer population individuals"""
        layer_fitnesses = []
        # includes 4 lists, each with layer_pop_size lists. Index here corresponds
        # to index for each ind in layer population, not its order.
        for i in range(self.n_layer_species):
            layer_species = []
            for j in range(self.layer_pop_size):
                layer_species.append([])
            layer_fitnesses.append(layer_species)
        for net_ind in net_pop:
            fit = net_ind.rank
            net_params = self.get_net_params(net_ind)
            for i in range(self.n_layer_species):
                layer_fitnesses[i][net_params['h' + str(i + 1)]].append(fit)
        for i, layer_species in enumerate(self.layer_population):
            for layer_ind in layer_species:
                fits = layer_fitnesses[i][layer_ind.index]
                if len(fits) > 0:
                    fits.sort()
                    if len(fits) > 1:
                        if fits[1] < 100:
                            # avg_fit = np.average(fits[:2]) # avg of top 2
                            avg_fit = fits[0]
                        else:
                            avg_fit = fits[0]
                    else:
                        avg_fit = fits[0]
                    min_fit = fits[0]
                    layer_ind.fitness.values = avg_fit, min_fit

    def cx_net_layers(self, net_ind_1, net_ind_2):
        """Uniform crossover of two networks layers"""
        # List of layers to swap. 0 is the first layer, and so on.
        idxs = list(set(np.random.randint(0, high=self.n_layer_species, size=self.n_layer_species)))
        for idx in idxs:
            sidx = self.s_idx_net[self.pos + idx]
            eidx = self.e_idx_net[self.pos + idx]
            container = net_ind_1[sidx:eidx]
            net_ind_1[sidx:eidx] = net_ind_2[sidx:eidx]
            net_ind_2[sidx:eidx] = container

    def mut_structure(self, net_ind):
        """mutate one layer index"""
        mut_gene = np.random.randint(self.pos, self.pos + self.n_layer_species)
        sidx = self.s_idx_net[mut_gene]
        eidx = self.e_idx_net[mut_gene]
        mut_bit = np.random.randint(sidx, eidx)
        net_ind[mut_bit] = int(not net_ind[mut_bit])

    def mut_parameters(self, net_ind):
        """Mutate network parameters"""
        net_fits = []
        for ind in self.net_population:
            net_fits.append(ind.rank)
        min_fit = np.min(net_fits)
        max_fit = np.max(net_fits)
        ind_fit = net_ind.rank
        # scale the fitness in a way that n_genes_mutated is between 0 and pos
        scaled_fit = (ind_fit - min_fit) / (max_fit - min_fit) * np.log(self.pos)
        n_genes_mutated = int(np.exp(scaled_fit))
        mut_genes = list(np.random.choice(np.arange(self.pos), n_genes_mutated, replace=False))
        for gene in mut_genes:
            sidx = self.s_idx_net[gene]
            eidx = self.e_idx_net[gene]
            mut_bit = np.random.randint(sidx, eidx)
            net_ind[mut_bit] = int(not net_ind[mut_bit])

    def mut_layer_ind(self, layer_ind, layer_species):
        """Mutate layer parameters"""
        ind_fit = layer_ind.fitness.values[1]
        fits = []
        for ind in layer_species:
            fits.append(ind.fitness.values[1])
        min_fit = min(fits)
        max_fit = max(fits)
        n_genes = len(self.layer_gene_bits)
        if min_fit != max_fit:
            # scale the fitness in a way that n_genes_mutated is between 0 and n_genes
            scaled_fit = (ind_fit - min_fit) / (max_fit - min_fit) * np.log(n_genes)
            n_genes_mutated = int(np.exp(scaled_fit))
        else:
            n_genes_mutated = 1
        mut_genes = list(np.random.choice(np.arange(n_genes), n_genes_mutated, replace=False))
        for gene in mut_genes:
            sidx = self.s_idx_layer[gene]
            eidx = self.e_idx_layer[gene]
            mut_bit = np.random.randint(sidx, eidx)
            layer_ind[mut_bit] = int(not layer_ind[mut_bit])

    def order_net_pop(self, net_pop):
        """Order network population first based of NSGA, and then each front based of Rho_MK"""
        fronts = self.toolbox.selectNSGA2fronts(net_pop, k=self.net_pop_size)
        for i in range(len(fronts)):
            fronts[i].sort(key=lambda x: x.fitness.values[0], reverse=True)
        ordered_net_pop = []
        for front in fronts:
            for ind in front:
                ordered_net_pop.append(ind)
        for i in range(len(ordered_net_pop)):
            ordered_net_pop[i].rank = i
        return ordered_net_pop

import os
import time
import datetime
import numpy as np
import random as rn
import matplotlib.pyplot as plt
import tools as tools_modified
from deap import base, creator, tools
from train import auto_encoder

# global random seeds
# (not to be confused with NN weight initialization random seed)
np.random.seed(0)
rn.seed(0)

start_time = time.time()

# Variables
# Number of hidden layers fixed to 4
n_layer_species = 4
# Size of layer population needs to be a power of 2
pop_size_bits = 6
layer_pop_size = 2 ** pop_size_bits
# Size of network population needs to be bigger than layer population,
# so that most of the layers participate in networks
net_pop_size = 80
# Number of generations
n_gens = 50
# AE training iterations
iters = 1000

# number of bits allocated for each layer chromosome gene
layer_gene_bits = {'L2': 8,
                   'SP': 8,
                   'SR': 8,
                   'act': 2,
                   'init': 2,
                   'n_neuron': 3}

# number of bits allocated for each network chromosome gene
net_gene_bits = {'batch': 3,
                 'optim': 2,
                 'learn_rate': 8,
                 'decay': 8,
                 'mom': 5,
                 'rand_seed': 6,
                 'h1': pop_size_bits,
                 'h2': pop_size_bits,
                 'h3': pop_size_bits,
                 'h4': pop_size_bits}  # layer genes always at the end of the chromosome

# range of values for each layer chromosome gene
layer_gene_range = [{'L2': np.logspace(-8, 0, 2 ** layer_gene_bits['L2'], dtype='float32'),
                     'SP': np.linspace(0.001, 0.1, 2 ** layer_gene_bits['SP'], dtype='float32'),
                     'SR': np.logspace(-3, 1, 2 ** layer_gene_bits['SR'], dtype='float32'),
                     'act': ['sigmoid', 'tanh', 'softsign', 'selu'],
                     'init': ['glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
                     'n_neuron': np.arange(3, 11)}]

for i in range(n_layer_species - 1):
    dic = layer_gene_range[0].copy()
    layer_gene_range.append(dic)

layer_gene_range[1]['n_neuron'] = np.arange(14, 29, 2)
layer_gene_range[2]['n_neuron'] = np.arange(20, 91, 10)
layer_gene_range[3]['n_neuron'] = np.arange(50, 121, 10)

# range of values for each network chromosome gene
net_gene_range = {'batch': [2 ** i for i in range(8)],
                  'optim': ['adam', 'nadam', 'rmsprop', 'adadelta'],
                  'learn_rate': np.logspace(-8, 1, 2 ** net_gene_bits['learn_rate'], dtype='float32'),
                  'decay': np.logspace(-8, 0, 2 ** net_gene_bits['decay'], dtype='float32'),
                  'mom': np.logspace(-0.05, -1e-5, 2 ** net_gene_bits['mom'], dtype='float32'),
                  'rand_seed': np.arange(2 ** net_gene_bits['rand_seed']),
                  'h1': [i for i in range(2 ** net_gene_bits['h1'])],
                  'h2': [i for i in range(2 ** net_gene_bits['h2'])],
                  'h3': [i for i in range(2 ** net_gene_bits['h3'])],
                  'h4': [i for i in range(2 ** net_gene_bits['h4'])]}

# In this version of Python, dictionary is ordered
s_idx_layer = []  # start indexes for layer chromosome genes
e_idx_layer = []  # end indexes for layer chromosome genes
s = 0
for i in layer_gene_bits:
    s_idx_layer.append(s)
    e_idx_layer.append(s + layer_gene_bits[i])
    s = s + layer_gene_bits[i]

s_idx_net = []  # start indexes for network chromosome genes
e_idx_net = []  # end indexes for network chromosome genes
s = 0
for i in net_gene_bits:
    s_idx_net.append(s)
    e_idx_net.append(s + net_gene_bits[i])
    s = s + net_gene_bits[i]

# chromosome lengths
layer_ind_size = sum(layer_gene_bits.values())
net_ind_size = sum(net_gene_bits.values())

# position of the first layer gene in net ind
pos = len(net_gene_bits) - n_layer_species

layer_weights = (-1, -1)  # avg, min
net_weights = (1, -1)  # Rho_MK, ValLoss

toolbox = base.Toolbox()

# DEAP library classes definition
creator.create('FitnessLayer', base.Fitness, weights=layer_weights)
creator.create('FitnessNet', base.Fitness, weights=net_weights)
creator.create('LayerIndividual', list, fitness=creator.FitnessLayer)
creator.create('NetIndividual', list, fitness=creator.FitnessNet)


def get_net_params(net_ind):
    """Returns a dictionary of the parameters corresponding to a network chromosome"""
    net_ind_str = str(net_ind).strip('[]').replace(' ', '').replace(',', '').replace('\n', '')
    net_ind_decimal = []
    for i in range(len(net_gene_bits.keys())):
        net_ind_decimal.append(int(net_ind_str[s_idx_net[i]:e_idx_net[i]], 2))
    net_params = {}
    for i, key in enumerate(net_gene_range):
        net_params[key] = net_gene_range[key][net_ind_decimal[i]]
    return net_params


def get_layer_params(layer_ind, species_idx):
    """Returns a dictionary of the parameters corresponding to a layer chromosome"""
    layer_ind_str = str(layer_ind).strip('[]').replace(' ', '').replace(',', '').replace('\n', '')
    layer_ind_decimal = []
    for i in range(len(layer_gene_bits.keys())):
        layer_ind_decimal.append(int(layer_ind_str[s_idx_layer[i]:e_idx_layer[i]], 2))
    layer_params = {}
    for i, key in enumerate(layer_gene_range[species_idx]):
        layer_params[key] = layer_gene_range[species_idx][key][layer_ind_decimal[i]]
    return layer_params


def eval_fitness(net_ind):
    """Calculates the fitness of a network individual
    Input: network individual
    Returns: a tuple of fitness values
    """
    net_params = get_net_params(net_ind)
    layer_params_list = []
    for i in range(n_layer_species):
        layer_species = layer_population[i].copy()
        layer_species.sort(key=lambda x: x.index)
        layer_ind = layer_species[net_params['h' + str(i + 1)]]
        layer_params = get_layer_params(layer_ind, i)
        layer_params_list.append(layer_params)
    net_ind.net_params = net_params
    net_ind.layer_params = layer_params_list
    return auto_encoder(net_params, layer_params_list,
                        n_layers=n_layer_species, iters=iters)


def layers_credit_assignment(net_pop):
    """Assigns credits to layer population individuals"""
    layer_fitnesses = []
    # includes 4 lists, each with layer_pop_size lists. Index here corresponds
    # to index for each ind in layer population, not its order.
    for i in range(n_layer_species):
        layer_species = []
        for j in range(layer_pop_size):
            layer_species.append([])
        layer_fitnesses.append(layer_species)
    for net_ind in net_pop:
        fit = net_ind.rank
        net_params = get_net_params(net_ind)
        for i in range(n_layer_species):
            layer_fitnesses[i][net_params['h' + str(i + 1)]].append(fit)
    for i, layer_species in enumerate(layer_population):
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


def rand_bin():
    return rn.randint(0, 1)


def cx_net_layers(net_ind_1, net_ind_2):
    """Uniform crossover of two networks layers"""
    # List of layers to swap. 0 is the first layer, and so on.
    idxs = list(set(np.random.randint(0, high=n_layer_species, size=n_layer_species)))
    for idx in idxs:
        sidx = s_idx_net[pos + idx]
        eidx = e_idx_net[pos + idx]
        container = net_ind_1[sidx:eidx]
        net_ind_1[sidx:eidx] = net_ind_2[sidx:eidx]
        net_ind_2[sidx:eidx] = container


def mut_structure(net_ind):
    """mutate one layer index"""
    mut_gene = np.random.randint(pos, pos + n_layer_species)
    sidx = s_idx_net[mut_gene]
    eidx = e_idx_net[mut_gene]
    mut_bit = np.random.randint(sidx, eidx)
    net_ind[mut_bit] = int(not net_ind[mut_bit])


def mut_parameters(net_ind):
    """Mutate network parameters"""
    net_fits = []
    for ind in net_population:
        net_fits.append(ind.rank)
    min_fit = np.min(net_fits)
    max_fit = np.max(net_fits)
    ind_fit = net_ind.rank
    # scale the fitness in a way that n_genes_mutated is between 0 and pos
    scaled_fit = (ind_fit - min_fit) / (max_fit - min_fit) * np.log(pos)
    n_genes_mutated = int(np.exp(scaled_fit))
    mut_genes = list(np.random.choice(np.arange(pos), n_genes_mutated, replace=False))
    for gene in mut_genes:
        sidx = s_idx_net[gene]
        eidx = e_idx_net[gene]
        mut_bit = np.random.randint(sidx, eidx)
        net_ind[mut_bit] = int(not net_ind[mut_bit])


def mut_layer_ind(layer_ind, layer_species):
    """Mutate layer parameters"""
    ind_fit = layer_ind.fitness.values[1]
    fits = []
    for ind in layer_species:
        fits.append(ind.fitness.values[1])
    min_fit = min(fits)
    max_fit = max(fits)
    n_genes = len(layer_gene_bits)
    if min_fit != max_fit:
        # scale the fitness in a way that n_genes_mutated is between 0 and n_genes
        scaled_fit = (ind_fit - min_fit) / (max_fit - min_fit) * np.log(n_genes)
        n_genes_mutated = int(np.exp(scaled_fit))
    else:
        n_genes_mutated = 1
    mut_genes = list(np.random.choice(np.arange(n_genes), n_genes_mutated, replace=False))
    for gene in mut_genes:
        sidx = s_idx_layer[gene]
        eidx = e_idx_layer[gene]
        mut_bit = np.random.randint(sidx, eidx)
        layer_ind[mut_bit] = int(not layer_ind[mut_bit])


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
        selection = selection | set([int(rank)])
    return [individuals[i] for i in selection]


def order_net_pop(net_pop, k=net_pop_size):
    """Order network population first based of NSGA, and then each front based of Rho_MK"""
    fronts = toolbox.selectNSGA2fronts(net_pop, k=k)
    for i in range(len(fronts)):
        fronts[i].sort(key=lambda x: x.fitness.values[0], reverse=True)
    ordered_net_pop = []
    for front in fronts:
        for ind in front:
            ordered_net_pop.append(ind)
    for i in range(len(ordered_net_pop)):
        ordered_net_pop[i].rank = i
    return ordered_net_pop


toolbox.register('rand_bin', rand_bin)
toolbox.register('layerIndividual', tools.initRepeat, creator.LayerIndividual,
                 toolbox.rand_bin, n=layer_ind_size)
toolbox.register('netIndividual', tools.initRepeat, creator.NetIndividual,
                 toolbox.rand_bin, n=net_ind_size)
toolbox.register('layerSpecies', tools.initRepeat, list, toolbox.layerIndividual,
                 n=layer_pop_size)
toolbox.register('netPopulation', tools.initRepeat, list, toolbox.netIndividual,
                 n=net_pop_size)
toolbox.register('evaluateNet', eval_fitness)
toolbox.register('mate', cx_net_layers)
toolbox.register('mutateNetStructure', mut_structure)
toolbox.register('mutateNetParameters', mut_parameters)
toolbox.register('mutateLayerParameters', mut_layer_ind)
toolbox.register("selectNSGA2fronts", tools_modified.selNSGA2, nd='standard')
toolbox.register("selectNSGA2", tools.selNSGA2, nd='standard')
toolbox.register("selectRoulette", sel_rank_roulette)


# initialize layer population
layer_population = [toolbox.layerSpecies() for i in range(n_layer_species)]
# assign idx for each layer individual
for layer_species in layer_population:
    for i, layer_ind in enumerate(layer_species):
        layer_ind.index = i
# initialize net population
net_population = toolbox.netPopulation()
# All initial layers should participate in networks
for i in range(layer_pop_size):
    bin_layer_idx = np.binary_repr(i, pop_size_bits)
    for j in range(n_layer_species):
        sidx = s_idx_net[j + pos]
        eidx = e_idx_net[j + pos]
        net_population[i][sidx:eidx] = list(np.int_(list(bin_layer_idx)))
for ind in net_population:
    ind.age = 0
print('\nEvaluating Initial Population...\n')
# evaluate initial network population
fits = toolbox.map(toolbox.evaluateNet, net_population)
for fit, ind in zip(fits, net_population):
    ind.fitness.values = fit
# order networks population
net_population = order_net_pop(net_population)
layers_credit_assignment(net_population)
# order layers population
for i in range(len(layer_population)):
    layer_population[i] = toolbox.selectNSGA2(layer_population[i], k=layer_pop_size)
print('\nInitial Population Fitness Values:\n')
for ind in net_population:
    print(ind.fitness.values)

# for saving populations and fitnesses
avg_layer_fits, avg_net_rho, min_layer_fits, max_net_rho = [], [], [], []
net_pops, layer_pops = [], []

initial_net_pop = toolbox.clone(net_population)
initial_layer_pop = toolbox.clone(layer_population)
for i in range(n_layer_species):
    initial_layer_pop[i].sort(key=lambda x: x.index)
net_pops.append(initial_net_pop)
layer_pops.append(initial_layer_pop)

avg_species_fits = []
min_species_fits = []
for spec in layer_population:
    fits = []
    for ind in spec:
        fits.append(ind.fitness.values[1])
    avg_species_fits.append(np.average(fits))
    min_species_fits.append(np.min(fits))
avg_layer_fits.append(avg_species_fits)
min_layer_fits.append(min_species_fits)

rhos = []
for ind in net_population:
    rhos.append(ind.fitness.values[0])
avg_net_rho.append(np.average(rhos))
max_net_rho.append(np.max(rhos))


for gen in range(n_gens):
    print('\nGeneration: ', gen + 1)
    elapsed_time = time.time() - start_time
    print('Elapsed Time: ', elapsed_time / 60, ' minutes\n')
    # network population evolution
    # structural and parametric mutation of diverged networks
    for ind in net_population:
        if ind.fitness.values[1] == 100.0:
            toolbox.mutateNetStructure(ind)
            toolbox.mutateNetParameters(ind)
            del ind.fitness.values
    n_net_top_inds = int(0.7 * net_pop_size)  # number of top 70% network individuals
    old_reached = False
    # used for calculating mutation probabilities
    fits = []
    for ind in net_population:
        fits.append(ind.rank)
    min_fit = min(fits)
    max_fit = max(fits)
    offspring = []
    idxs = list(np.random.choice(np.arange(n_net_top_inds), size=2, replace=False))
    for idx in idxs:
        child = toolbox.clone(net_population[idx])
        toolbox.mutateNetParameters(child)
        child.age = 0
        offspring.append(child)
    bottom_inds = toolbox.clone(net_population[n_net_top_inds:])
    # Roulette selection of 2 networks from bottom 30%
    children = toolbox.selectRoulette(bottom_inds, k=2)
    toolbox.mate(children[0], children[1])
    for i in range(len(children)):
        mut_pb = np.sqrt((children[i].rank - min_fit) / (max_fit - min_fit))
        if np.random.random() < mut_pb:
            toolbox.mutateNetStructure(children[i])
        if np.random.random() < mut_pb:
            toolbox.mutateNetParameters(children[i])
        children[i].age = 0
    for child in children:
        offspring.append(child)
    for ind in offspring:
        del ind.fitness.values
    # delete worst len(offspring)-1 networks
    for i in range(len(offspring) - 1):
        del net_population[-1]
    for ind in net_population:
        ind.age += 1
        if ind.age >= 10:
            old_reached = True
    # for now we don't delete the old network
    old_reached = False
    # delete the oldest network if old reached, else delete worst network
    if old_reached:
        net_population.sort(key=lambda x: x.age)
        del net_population[-1]
    else:
        del net_population[-1]
    for i in range(len(offspring)):
        net_population.append(offspring[i])
    # layer population evolution
    n_layer_top_inds = int(0.7 * layer_pop_size)  # number of top 70% layer individuals
    # delete worst 30% of layers
    deleted_indexes = []
    for i, species in enumerate(layer_population):
        del_indexes = []
        for ind in species[n_layer_top_inds:]:
            del_indexes.append(ind.index)
        deleted_indexes.append(del_indexes)
        del species[n_layer_top_inds:]
        remaining_indexes = []
        for ind in species:
            remaining_indexes.append(ind.index)
        # random selection of layers to mutate
        new_idxs = list(np.random.choice(remaining_indexes,
                                         size=layer_pop_size - n_layer_top_inds,
                                         replace=False))
        assert len(del_indexes) == len(new_idxs)
        for j, idx in enumerate(new_idxs):
            child = toolbox.clone([ind for ind in species if ind.index == idx][0])
            toolbox.mutateLayerParameters(child, species)
            # assign the index of one of the deleted layers to the new one
            child.index = del_indexes[j]
            species.append(child)
    # evaluation
    for ind in net_population:
        for i in range(n_layer_species):
            if ind.net_params['h{}'.format(i + 1)] in deleted_indexes[i]:
                del ind.fitness.values
    invalid_net_inds = [ind for ind in net_population if not ind.fitness.valid]
    fits = map(toolbox.evaluateNet, invalid_net_inds)
    for ind, fit in zip(invalid_net_inds, fits):
        ind.fitness.values = fit
    net_population = order_net_pop(net_population)
    layers_credit_assignment(net_population)
    for i in range(len(layer_population)):
        layer_population[i] = toolbox.selectNSGA2(layer_population[i], k=layer_pop_size)
    # saving fitnesses
    avg_species_fits = []
    min_species_fits = []
    for spec in layer_population:
        fits = []
        for ind in spec:
            fits.append(ind.fitness.values[1])
        avg_species_fits.append(np.average(fits))
        min_species_fits.append(np.min(fits))
    avg_layer_fits.append(avg_species_fits)
    min_layer_fits.append(min_species_fits)
    rhos = []
    for ind in net_population:
        rhos.append(ind.fitness.values[0])
    avg_net_rho.append(np.average(rhos))
    max_net_rho.append(np.max(rhos))
    # saving populations
    pop = toolbox.clone(layer_population)
    for i in range(n_layer_species):
        pop[i].sort(key=lambda x: x.index)
    layer_pops.append(pop)
    pop = toolbox.clone(net_population)
    net_pops.append(pop)
    del pop
    # printing network population fitnesses
    for ind in net_population:
        print(ind.fitness.values)


end_time = time.time()
run_time = end_time - start_time
print('\nTotal run time is:', run_time / 3600, 'hours')


date = str(datetime.date.today())
time = str(datetime.datetime.now().time())[:8].replace(':', '-')
dirr = 'results/{}/{}'.format(date, time)

os.makedirs(dirr + '/pops')

plt.figure()
plt.plot(max_net_rho)
plt.xlabel('Generations')
plt.ylabel('Rho_MK')
plt.title('Max Rho_MK per generation')
plt.savefig(dirr + '/' + 'Max_Rho_MK_per_generation.png')

plt.figure()
plt.plot(avg_net_rho)
plt.xlabel('Generations')
plt.ylabel('Rho_MK')
plt.title('Average Rho_MK per generation')
plt.savefig(dirr + '/' + 'Average_Rho_MK_per_generation.png')

for i in range(n_layer_species):
    plt.figure()
    plt.plot(np.array(avg_layer_fits)[:, i])
    plt.xlabel('Generation')
    plt.ylabel('"Min" Fitness')
    plt.title('Average "Min" Fitness of the Layer Species nr. {} Per Generation'.format(i))
    plt.savefig(dirr + '/' + 'Average_Min_Fitness_of_the_Layer_Species_nr_{}_Per_Generation.png'.format(i))

for i, pop in enumerate(net_pops):
    xs, ys = [], []
    for ind in pop:
        xs.append(ind.fitness.values[0])
        ys.append(ind.fitness.values[1])
    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel('Rho_MK')
    plt.ylabel('val_loss')
    plt.title('Generation {} front'.format(i))
    plt.savefig(dirr + '/pops/' + 'Generation_{}_population.png'.format(i))


# hash codes of strings of chromosomes
net_pop_ids = []
for pop in net_pops:
    lis = []
    for ind in pop:
        lis.append(hash(str(ind)))
    net_pop_ids.append(lis)

# index of each chromosome of each generation population in the next generation population
indexes = toolbox.clone(net_pop_ids)
for i in range(len(net_pops) - 1):
    for j in range(net_pop_size):
        _ = np.equal(net_pop_ids[i][j], net_pop_ids[i + 1])
        if bool(np.isin(True, _)):
            indexes[i][j] = list(_).index(True)
        else:
            indexes[i][j] = None
for k in range(net_pop_size):
    indexes[-1][k] = None

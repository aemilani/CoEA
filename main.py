import numpy as np
import random as rn
from train import autoEncoder
from deap import base, creator, tools
import matplotlib.pyplot as plt
import time
import tools as tools_modified
import datetime
import os

# global random seeds
# (not to be confused with NN weight initialization random seed)
np.random.seed(0)
rn.seed(0)

startTime = time.time()

# Variables
# Number of hidden layers fixed to 4
nLayerSpecies = 4
# Size of layer population needs to be a power of 2
popSizeBits = 6
layerPopSize = 2**popSizeBits
# Size of network population needs to be bigger than layer population,
# so that most of the layers participate in networks
netPopSize = 80
# Number of generations
nGens = 1
# AE training iterations
iters = 1000

# number of bits allocated for each layer chromosome gene
layerGeneBits = {'L2':8,
                 'SP':8,
                 'SR':8,
                 'act':2,
                 'init':2,
                 'n_neuron':3}

# number of bits allocated for each network chromosome gene
netGeneBits = {'batch':3,
               'optim':2,
               'learn_rate':8,
               'decay':8,
               'mom':5,
               'rand_seed':6,
               'h1':popSizeBits,
               'h2':popSizeBits,
               'h3':popSizeBits,
               'h4':popSizeBits} #layer genes always at the end of the chromosome

# range of values for each layer chromosome gene
layerGeneRange = [{'L2':np.logspace(-8, 0, 2**layerGeneBits['L2'], dtype='float32'),
                 'SP':np.linspace(0.001, 0.1, 2**layerGeneBits['SP'], dtype='float32'),
                 'SR':np.logspace(-3, 1, 2**layerGeneBits['SR'], dtype='float32'),
                 'act':['sigmoid','tanh','softsign','selu'],
                 'init':['glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
                 'n_neuron':np.arange(3, 11)}]

for i in range(nLayerSpecies-1):
    dic = layerGeneRange[0].copy()
    layerGeneRange.append(dic)

layerGeneRange[1]['n_neuron'] = np.arange(14, 29, 2)
layerGeneRange[2]['n_neuron'] = np.arange(20, 91, 10)
layerGeneRange[3]['n_neuron'] = np.arange(50, 121, 10)

# range of values for each network chromosome gene
netGeneRange = {'batch':[2**i for i in range(8)],
                'optim':['adam','nadam','rmsprop','adadelta'],
                'learn_rate':np.logspace(-8, 1, 2**netGeneBits['learn_rate'], dtype='float32'),
                'decay':np.logspace(-8, 0, 2**netGeneBits['decay'], dtype='float32'),
                'mom':np.logspace(-0.05, -1e-5, 2**netGeneBits['mom'], dtype='float32'),
                'rand_seed':np.arange(2**netGeneBits['rand_seed']),
                'h1':[i for i in range(2**netGeneBits['h1'])],
                'h2':[i for i in range(2**netGeneBits['h2'])],
                'h3':[i for i in range(2**netGeneBits['h3'])],
                'h4':[i for i in range(2**netGeneBits['h4'])]}

# In this version of Python, dictionary is ordered
sidxLayer = [] # start indexes for layer chromosome genes
eidxLayer = [] # end indexes for layer chromosome genes
s = 0
for i in layerGeneBits:
    sidxLayer.append(s)
    eidxLayer.append(s + layerGeneBits[i])
    s = s + layerGeneBits[i]
    
sidxNet = [] # start indexes for network chromosome genes
eidxNet = [] # end indexes for network chromosome genes
s = 0
for i in netGeneBits:
    sidxNet.append(s)
    eidxNet.append(s + netGeneBits[i])
    s = s + netGeneBits[i]

# chromosome lengths
layerIndSize = sum(layerGeneBits.values())
netIndSize = sum(netGeneBits.values())

# position of the first layer gene in net ind
pos = len(netGeneBits) - nLayerSpecies

layerWeights = (-1, -1) # avg, min
netWeights = (1, -1) # Rho_MK, ValLoss

#%%
# DEAP library classes definition
creator.create('FitnessLayer', base.Fitness, weights = layerWeights)
creator.create('FitnessNet', base.Fitness, weights = netWeights)
creator.create('LayerIndividual', list, fitness = creator.FitnessLayer)
creator.create('NetIndividual', list, fitness = creator.FitnessNet)

def getNetParams(netInd):
    """Returns a dictionary of the parameters corresponding to a network chromosome"""
    netIndStr = str(netInd).strip('[]').replace(' ', '').replace(',', '').replace('\n','')
    netIndDecimal = []
    for i in range(len(netGeneBits.keys())):
        netIndDecimal.append(int(netIndStr[sidxNet[i]:eidxNet[i]], 2))
    netParams = {}
    for i, key in enumerate(netGeneRange):
        netParams[key] = netGeneRange[key][netIndDecimal[i]]
    return netParams

def getLayerParams(layerInd, speciesIdx):
    """Returns a dictionary of the parameters corresponding to a layer chromosome"""
    layerIndStr = str(layerInd).strip('[]').replace(' ', '').replace(',', '').replace('\n','')
    layerIndDecimal = []
    for i in range(len(layerGeneBits.keys())):
        layerIndDecimal.append(int(layerIndStr[sidxLayer[i]:eidxLayer[i]], 2))
    layerParams = {}
    for i, key in enumerate(layerGeneRange[speciesIdx]):
        layerParams[key] = layerGeneRange[speciesIdx][key][layerIndDecimal[i]]
    return layerParams

def evalFitness(netInd):
    """Calculates the fitness of a network individual
    Input: network individual
    Returns: a tuple of fitness values
    """
    netParams = getNetParams(netInd)
    layerParamsList = []
    for i in range(nLayerSpecies):
        layerSpecies = layerPopulation[i].copy()
        layerSpecies.sort(key=lambda x: x.index)
        layerInd = layerSpecies[netParams['h'+str(i+1)]]
        layerParams = getLayerParams(layerInd, i)
        layerParamsList.append(layerParams)
    netInd.net_params = netParams
    netInd.layer_params = layerParamsList
    return autoEncoder(netParams, layerParamsList,
                       nLayers=nLayerSpecies, iters=iters)

def layersCreditAssignment(netPop):
    """Assigns credits to layer population individuals"""
    layerFitnesses = []
    # includes 4 lists, each with layerPopSize lists. Index here corresponds
    # to index for each ind in layer population, not its order.
    for i in range(nLayerSpecies):
        layerSpecies = []
        for j in range(layerPopSize):
            layerSpecies.append([])
        layerFitnesses.append(layerSpecies)
    for netInd in netPop:
        fit = netInd.rank
        netParams = getNetParams(netInd)
        for i in range(nLayerSpecies):
            layerFitnesses[i][netParams['h'+str(i+1)]].append(fit)
    for i, layerSpecies in enumerate(layerPopulation):
        for layerInd in layerSpecies:
            fits = layerFitnesses[i][layerInd.index]
            if len(fits) > 0:
                fits.sort()
                if len(fits) > 1:
                    if fits[1] < 100:
                        #avgFit = np.average(fits[:2]) # avg of top 2
                        avgFit = fits[0]
                    else:
                        avgFit = fits[0]
                else:
                    avgFit = fits[0]
                minFit = fits[0]
                layerInd.fitness.values = avgFit, minFit

def rand_bin():
    return rn.randint(0,1)

def cxNetLayers(netInd1, netInd2):
    """Uniform crossover of two networks layers"""
    # List of layers to swap. 0 is the first layer, and so on.
    idxs = list(set(np.random.randint(0, high=nLayerSpecies, size=nLayerSpecies)))
    for idx in idxs:
        sidx = sidxNet[pos+idx]
        eidx = eidxNet[pos+idx]
        container = netInd1[sidx:eidx]
        netInd1[sidx:eidx] = netInd2[sidx:eidx]
        netInd2[sidx:eidx] = container

def mutStructure(netInd):
    """mutate one layer index"""
    mutGene = np.random.randint(pos, pos+nLayerSpecies)
    sidx = sidxNet[mutGene]
    eidx = eidxNet[mutGene]
    mutBit = np.random.randint(sidx, eidx)
    netInd[mutBit] = int(not netInd[mutBit])

def mutParameters(netInd):
    """Mutate network parameters"""
    netFits = []
    for ind in netPopulation:
        netFits.append(ind.rank)
    minFit = np.min(netFits)
    maxFit = np.max(netFits)
    indFit = netInd.rank
    # scale the fitness in a way that nGenesMutated is between 0 and pos
    scaledFit = (indFit - minFit) / (maxFit - minFit) * np.log(pos)
    nGenesMutated = int(np.exp(scaledFit))
    mutGenes = list(np.random.choice(np.arange(pos), nGenesMutated, replace=False))
    for gene in mutGenes:
        sidx = sidxNet[gene]
        eidx = eidxNet[gene]
        mutBit = np.random.randint(sidx, eidx)
        netInd[mutBit] = int(not netInd[mutBit])

def mutLayerInd(layerInd, layerSpecies):
    """Mutate layer parameters"""
    indFit = layerInd.fitness.values[1]
    fits =[]
    for ind in layerSpecies:
        fits.append(ind.fitness.values[1])
    minFit = min(fits)
    maxFit = max(fits)
    nGenes = len(layerGeneBits)
    if minFit != maxFit:
        # scale the fitness in a way that nGenesMutated is between 0 and nGenes
        scaledFit = (indFit - minFit) / (maxFit - minFit) * np.log(nGenes)
        nGenesMutated = int(np.exp(scaledFit))
    else:
        nGenesMutated = 1
    mutGenes = list(np.random.choice(np.arange(nGenes), nGenesMutated, replace=False))
    for gene in mutGenes:
        sidx = sidxLayer[gene]
        eidx = eidxLayer[gene]
        mutBit = np.random.randint(sidx, eidx)
        layerInd[mutBit] = int(not layerInd[mutBit])
        
def selRankRoulette(individuals, k=2):
    """Rank-based Roulette selection"""
    fits = []
    for ind in individuals:
        fits.append(1/(ind.rank+1))
    fitsSum = sum(fits)
    fits = [fits[i]/fitsSum for i in range(len(fits))]
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

def orderNetPop(netPop, k=netPopSize):
    """Order network population first based of NSGA, and then each front based of Rho_MK"""
    fronts = toolbox.selectNSGA2fronts(netPop, k=k)
    for i in range(len(fronts)):
        fronts[i].sort(key=lambda x: x.fitness.values[0], reverse=True)
    orderedNetPop = []
    for front in fronts:
        for ind in front:
            orderedNetPop.append(ind)
    for i in range(len(orderedNetPop)):
        orderedNetPop[i].rank = i
    return orderedNetPop

toolbox = base.Toolbox()
toolbox.register('rand_bin', rand_bin)
toolbox.register('layerIndividual', tools.initRepeat, creator.LayerIndividual, \
                 toolbox.rand_bin, n=layerIndSize)
toolbox.register('netIndividual', tools.initRepeat, creator.NetIndividual, \
                 toolbox.rand_bin, n=netIndSize)
toolbox.register('layerSpecies', tools.initRepeat, list, toolbox.layerIndividual, \
                 n=layerPopSize)
toolbox.register('netPopulation', tools.initRepeat, list, toolbox.netIndividual, \
                 n=netPopSize)
toolbox.register('evaluateNet', evalFitness)
toolbox.register('mate', cxNetLayers)
toolbox.register('mutateNetStructure', mutStructure)
toolbox.register('mutateNetParameters', mutParameters)
toolbox.register('mutateLayerParameters', mutLayerInd)
toolbox.register("selectNSGA2fronts", tools_modified.selNSGA2, nd='standard')
toolbox.register("selectNSGA2", tools.selNSGA2, nd='standard')
toolbox.register("selectRoulette", selRankRoulette)

#%%
print('\nEvaluating Initial Population...\n')
# initialize layer population
layerPopulation = [toolbox.layerSpecies() for i in range(nLayerSpecies)]
# assign idx for each layer individual
for layer_species in layerPopulation:
    for i, layerInd in enumerate(layer_species):
        layerInd.index = i
# initialize net population
netPopulation = toolbox.netPopulation()
# All initial layers should participate in networks
for i in range(layerPopSize):
    binLayerIdx = np.binary_repr(i, popSizeBits)
    for j in range(nLayerSpecies):
        sidx = sidxNet[j+pos]
        eidx = eidxNet[j+pos]
        netPopulation[i][sidx:eidx] = list(np.int_(list(binLayerIdx)))
for ind in netPopulation:
    ind.age = 0
# evaluate initial network population
fits = toolbox.map(toolbox.evaluateNet, netPopulation)
for fit, ind in zip(fits, netPopulation):
    ind.fitness.values = fit
# order networks population
netPopulation = orderNetPop(netPopulation)
layersCreditAssignment(netPopulation)
# order layers population
for i in range(len(layerPopulation)):
    layerPopulation[i] = toolbox.selectNSGA2(layerPopulation[i], k=layerPopSize)
print('\nInitial Population Fitness Values:\n')
for ind in netPopulation:
    print(ind.fitness.values)

# for saving populations and fitnesses
avgLayerFits, avgNetRho, minLayerFits, maxNetRho = [], [], [], []
netPops, layerPops = [], []

initialNetPop = toolbox.clone(netPopulation)
initialLayerPop = toolbox.clone(layerPopulation)
for i in range(nLayerSpecies):
    initialLayerPop[i].sort(key=lambda x: x.index)
netPops.append(initialNetPop)
layerPops.append(initialLayerPop)

avgSpeciesFits = []
minSpeciesFits = []
for spec in layerPopulation:
    fits = []
    for ind in spec:
        fits.append(ind.fitness.values[1])
    avgSpeciesFits.append(np.average(fits))
    minSpeciesFits.append(np.min(fits))
avgLayerFits.append(avgSpeciesFits)
minLayerFits.append(minSpeciesFits)

rhos = []
for ind in netPopulation:
    rhos.append(ind.fitness.values[0])
avgNetRho.append(np.average(rhos))
maxNetRho.append(np.max(rhos))

#%%
for gen in range(nGens):
    print('\nGeneration: ', gen+1)
    elapsedTime = time.time() - startTime
    print('Elapsed Time: ', elapsedTime/60, ' minutes\n')
    # network population evolution
    # structural and parametric mutation of diverged networks
    for ind in netPopulation:
        if ind.fitness.values[1] == 100.0:
            toolbox.mutateNetStructure(ind)
            toolbox.mutateNetParameters(ind)
            del ind.fitness.values
    nNetTopInds = int(0.7*netPopSize) # number of top 70% network individuals
    oldReached = False
    # used for calculating mutation probabilities
    fits = []
    for ind in netPopulation:
        fits.append(ind.rank)
    minFit = min(fits)
    maxFit = max(fits)
    offspring = []
    idxs = list(np.random.choice(np.arange(nNetTopInds), size=2, replace=False))
    for idx in idxs:
        child = toolbox.clone(netPopulation[idx])
        toolbox.mutateNetParameters(child)
        child.age = 0
        offspring.append(child)
    bottomInds = toolbox.clone(netPopulation[nNetTopInds:])
    # Roulette selection of 2 networks from bottom 30%
    children = toolbox.selectRoulette(bottomInds, k=2)
    toolbox.mate(children[0], children[1])
    for i in range(len(children)):
        mutPb = np.sqrt((children[i].rank - minFit) / (maxFit - minFit))
        if np.random.random() < mutPb:
            toolbox.mutateNetStructure(children[i])
        if np.random.random() < mutPb:
            toolbox.mutateNetParameters(children[i])
        children[i].age = 0
    for child in children:
        offspring.append(child)
    for ind in offspring:
        del ind.fitness.values
    # delete worst len(offspring)-1 networks
    for i in range(len(offspring) - 1):
        del netPopulation[-1]
    for ind in netPopulation:
        ind.age += 1
        if ind.age >= 10:
            oldReached = True
    # for now we don't delete the old network
    oldReached = False
    # delete the oldest network if old reached, else delete worst network
    if oldReached:
        netPopulation.sort(key=lambda x: x.age)
        del netPopulation[-1]
    else:
        del netPopulation[-1]
    for i in range(len(offspring)):
        netPopulation.append(offspring[i])
    # layer population evolution
    nLayerTopInds = int(0.7*layerPopSize) # number of top 70% layer individuals
    # delete worst 30% of layers
    deleted_indexes = []
    for i, species in enumerate(layerPopulation):
        deletedIndexes = []
        for ind in species[nLayerTopInds:]:
            deletedIndexes.append(ind.index)
        deleted_indexes.append(deletedIndexes)
        del species[nLayerTopInds:]
        remainingIndexes = []
        for ind in species:
            remainingIndexes.append(ind.index)
        # random selection of layers to mutate
        newIdxs = list(np.random.choice(remainingIndexes,
                                     size=layerPopSize-nLayerTopInds,
                                     replace=False))
        assert len(deletedIndexes) == len(newIdxs)
        for j, idx in enumerate(newIdxs):
            child = toolbox.clone([ind for ind in species if ind.index==idx][0])
            toolbox.mutateLayerParameters(child, species)
            # assign the index of one of the deleted layers to the new one
            child.index = deletedIndexes[j]
            species.append(child)
    # evaluation
    for ind in netPopulation:
        for i in range(nLayerSpecies):
            if ind.net_params['h{}'.format(i+1)] in deleted_indexes[i]:
                del ind.fitness.values
    invalid_net_inds = [ind for ind in netPopulation if not ind.fitness.valid]
    fits = map(toolbox.evaluateNet, invalid_net_inds)
    for ind, fit in zip(invalid_net_inds, fits):
        ind.fitness.values = fit
    netPopulation = orderNetPop(netPopulation)
    layersCreditAssignment(netPopulation)
    for i in range(len(layerPopulation)):
        layerPopulation[i] = toolbox.selectNSGA2(layerPopulation[i], k=layerPopSize)
    # saving fitnesses
    avgSpeciesFits = []
    minSpeciesFits = []
    for spec in layerPopulation:
        fits = []
        for ind in spec:
            fits.append(ind.fitness.values[1])
        avgSpeciesFits.append(np.average(fits))
        minSpeciesFits.append(np.min(fits))
    avgLayerFits.append(avgSpeciesFits)
    minLayerFits.append(minSpeciesFits)
    rhos = []
    for ind in netPopulation:
        rhos.append(ind.fitness.values[0])
    avgNetRho.append(np.average(rhos))
    maxNetRho.append(np.max(rhos))
    # saving populations
    pop = toolbox.clone(layerPopulation)
    for i in range(nLayerSpecies):
        pop[i].sort(key= lambda x: x.index)
    layerPops.append(pop)
    pop = toolbox.clone(netPopulation)
    netPops.append(pop)
    del pop
    # printing network population fitnesses
    for ind in netPopulation:
        print(ind.fitness.values)
    
#%%
endTime = time.time()
runTime = endTime - startTime
print('\nTotal run time is:',runTime/3600,'hours')

#%%
date = str(datetime.date.today())
time = str(datetime.datetime.now().time())[:8].replace(':', '-')
dirr ='results/{}/{}'.format(date, time)

os.makedirs(dirr + '/pops')

plt.figure()
plt.plot(maxNetRho)
plt.xlabel('Generations')
plt.ylabel('Rho_MK')
plt.title('Max Rho_MK per generation')
plt.savefig(dirr + '/' + 'Max_Rho_MK_per_generation.png')

plt.figure()
plt.plot(avgNetRho)
plt.xlabel('Generations')
plt.ylabel('Rho_MK')
plt.title('Average Rho_MK per generation')
plt.savefig(dirr + '/' + 'Average_Rho_MK_per_generation.png')

for i in range(nLayerSpecies):
    plt.figure()
    plt.plot(np.array(avgLayerFits)[:,i])
    plt.xlabel('Generation')
    plt.ylabel('"Min" Fitness')
    plt.title('Average "Min" Fitness of the Layer Species nr. {} Per Generation'.format(i))
    plt.savefig(dirr + '/' + 'Average_Min_Fitness_of_the_Layer_Species_nr_{}_Per_Generation.png'.format(i))

for i, pop in enumerate(netPops):
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
    
#%%
# hash codes of strings of chromosomes
netPopIDs = []
for pop in netPops:
    lis = []
    for ind in pop:
        lis.append(hash(str(ind)))
    netPopIDs.append(lis)

# index of each chromosome of each generation population in the next generation population
indexes = toolbox.clone(netPopIDs)
for i in range(len(netPops)-1):
    for j in range(netPopSize):
        _ = np.equal(netPopIDs[i][j], netPopIDs[i+1])
        if bool(np.isin(True, _)):
            indexes[i][j] = list(_).index(True)
        else:
            indexes[i][j] = None
for k in range(netPopSize):
    indexes[-1][k] = None

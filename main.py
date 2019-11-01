import numpy as np
import random as rn
from train import autoEncoder
from deap import base, creator, tools
import matplotlib.pyplot as plt
import time

startTime = time.time()

# Variables
nLayerSpecies = 4
popSizeBits = 5
layerPopSize = 2**popSizeBits
netPopSize = 50
nGens = 60
iters = 4000 # AE iters

# Chromosome definition
layerGeneBits = {'L2':8,
                 'SP':8,
                 'SR':8,
                 'act':2,
                 'init':2,
                 'n_neuron':3}

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

# dictionary is ordered in this version of Python
sidxLayer = [] # start indexes for genes
eidxLayer = [] # end indexes for genes
s = 0
for i in layerGeneBits:
    sidxLayer.append(s)
    eidxLayer.append(s + layerGeneBits[i])
    s = s + layerGeneBits[i]
    
sidxNet = [] # start indexes for genes
eidxNet = [] # end indexes for genes
s = 0
for i in netGeneBits:
    sidxNet.append(s)
    eidxNet.append(s + netGeneBits[i])
    s = s + netGeneBits[i]

layerIndSize = sum(layerGeneBits.values())
netIndSize = sum(netGeneBits.values())

# position of the first layer gene in net ind
pos = len(netGeneBits) - nLayerSpecies

layerWeights = (-1, -1) # avg, min
netWeights = (1, -1) # Rho_MK, ValLoss

def hyperVolume(pop):
    mseMax = 100
    rhoMax = 20
    P = pop.copy()
    P.sort(key=lambda x: x.fitness.values[0])
    i = 0
    s = 0
    for ind in P:
        s += (ind.fitness.values[1] - i) * (mseMax - ind.fitness.values[0])
        i = ind.fitness.values[1]
    return s/(mseMax*rhoMax)

#%%

creator.create('FitnessLayer', base.Fitness, weights = layerWeights)
creator.create('FitnessNet', base.Fitness, weights = netWeights)
creator.create('LayerIndividual', list, fitness = creator.FitnessLayer)
creator.create('NetIndividual', list, fitness = creator.FitnessNet)

def getNetParams(netInd):
    netIndStr = str(netInd).strip('[]').replace(' ', '').replace(',', '').replace('\n','')
    netIndDecimal = []
    for i in range(len(netGeneBits.keys())):
        netIndDecimal.append(int(netIndStr[sidxNet[i]:eidxNet[i]], 2))
    netParams = {}
    for i, key in enumerate(netGeneRange):
        netParams[key] = netGeneRange[key][netIndDecimal[i]]
    return netParams

def getLayerParams(layerInd, speciesIdx):
    layerIndStr = str(layerInd).strip('[]').replace(' ', '').replace(',', '').replace('\n','')
    layerIndDecimal = []
    for i in range(len(layerGeneBits.keys())):
        layerIndDecimal.append(int(layerIndStr[sidxLayer[i]:eidxLayer[i]], 2))
    layerParams = {}
    for i, key in enumerate(layerGeneRange[speciesIdx]):
        layerParams[key] = layerGeneRange[speciesIdx][key][layerIndDecimal[i]]
    return layerParams

def evalFitness(netInd):
    netParams = getNetParams(netInd)
    layerParamsList = []
    for i in range(nLayerSpecies):
        layerSpecies = layerPopulation[i].copy()
        layerSpecies.sort(key=lambda x: x.index)
        layerInd = layerSpecies[netParams['h'+str(i+1)]]
        layerParams = getLayerParams(layerInd, i)
        layerParamsList.append(layerParams)
    return autoEncoder(netParams, layerParamsList,
                       nLayers=nLayerSpecies, iters=iters)

def layersCreditAssignment(netPop):
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
                        avgFit = np.average(fits[:2]) # avg of top 2
                        #avgFit = fits[0]
                    else:
                        avgFit = fits[0]
                else:
                    avgFit = fits[0]
                minFit = fits[0]
                layerInd.fitness.values = avgFit, minFit

def rand_bin():
    return rn.randint(0,1)

def cxNetLayers(netInd1, netInd2):
    idxs = list(set(np.random.randint(0, high=nLayerSpecies, size=nLayerSpecies)))
    for idx in idxs:
        sidx = sidxNet[pos+idx]
        eidx = eidxNet[pos+idx]
        container = netInd1[sidx:eidx]
        netInd1[sidx:eidx] = netInd2[sidx:eidx]
        netInd2[sidx:eidx] = container

def mutStructure(netInd):
    '''mutate one layer index'''
    mutGene = np.random.randint(pos, pos+nLayerSpecies)
    sidx = sidxNet[mutGene]
    eidx = eidxNet[mutGene]
    mutBit = np.random.randint(sidx, eidx)
    netInd[mutBit] = int(not netInd[mutBit])

def mutParameters(netInd):
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
    '''replace = False'''
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
toolbox.register("selectNSGA2", tools.selNSGA2, nd='standard')
toolbox.register("selectRoulette", selRankRoulette)

#%%
print()
print('Evaluating Initial Populations...')
# initialize layer population
layerPopulation = [toolbox.layerSpecies() for i in range(nLayerSpecies)]
# assign idx for each layer individual
for layer_species in layerPopulation:
    for i, layerInd in enumerate(layer_species):
        layerInd.index = i
# initialize net population
netPopulation = toolbox.netPopulation()
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
netPopulation = toolbox.selectNSGA2(netPopulation, k=netPopSize)
for i, ind in enumerate(netPopulation):
    ind.rank = i
layersCreditAssignment(netPopulation)
for i in range(len(layerPopulation)):
    layerPopulation[i] = toolbox.selectNSGA2(layerPopulation[i], k=layerPopSize)
print()
for ind in netPopulation:
    print(ind.fitness.values)
    
#%%
avgLayerFits, avgNetRho, minLayerFits, maxNetRho = [], [], [], []
#%%
for gen in range(nGens):
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
    print()
    print('Generation: ', gen+1)
    elapsedTime = time.time() - startTime
    print('Elapsed Time: ', elapsedTime/60, ' minutes')
    print()
    for ind in netPopulation:
        print(ind.fitness.values)
    # network population evolution
    for ind in netPopulation:
        if ind.fitness.values[1] == 100.0:
            toolbox.mutateNetStructure(ind)
            toolbox.mutateNetParameters(ind)
    nTopInds = int(0.7*netPopSize) # number of top 70% network individuals
    oldReached = False
    fits = []
    for ind in netPopulation:
        fits.append(ind.rank)
    minFit = min(fits)
    maxFit = max(fits)
    offspring = []
    idxs = list(np.random.choice(np.arange(nTopInds), size=2, replace=False))
    for idx in idxs:
        child = toolbox.clone(netPopulation[idx])
        toolbox.mutateNetParameters(child)
        child.age = 0
        offspring.append(child)
    bottomInds = toolbox.clone(netPopulation[nTopInds:])
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
    for i in range(len(offspring) - 1):
        del netPopulation[-1]
    for ind in netPopulation:
        ind.age += 1
        if ind.age >= 10:
            oldReached = True
    oldReached = False
    if oldReached:
        netPopulation.sort(key=lambda x: x.age)
        del netPopulation[-1]
    else:
        del netPopulation[-1]
    for i in range(len(offspring)):
        netPopulation.append(offspring[i])
    # layer population evolution
    nTopInds = int(0.7*layerPopSize) # number of top 70% layer individuals
    for species in layerPopulation:
        deletedIndexes = []
        for ind in species[nTopInds:]:
            deletedIndexes.append(ind.index)
        del species[nTopInds:]
        idxs = list(np.random.choice(np.arange(nTopInds),
                                     size=layerPopSize-nTopInds, replace=False))
        assert len(deletedIndexes) == len(idxs)
        for i, idx in enumerate(idxs):
            child = toolbox.clone(species[idx])
            toolbox.mutateLayerParameters(child, species)
            child.index = deletedIndexes[i]
            species.append(child)
    # evaluation
    fits = map(toolbox.evaluateNet, netPopulation)
    for ind, fit in zip(netPopulation, fits):
        ind.fitness.values = fit
    layersCreditAssignment(netPopulation)
    netPopulation = toolbox.selectNSGA2(netPopulation, k=netPopSize)
    for i, ind in enumerate(netPopulation):
        ind.rank = i
    for i in range(len(layerPopulation)):
        layerPopulation[i] = toolbox.selectNSGA2(layerPopulation[i], k=layerPopSize)
    
#%%
endTime = time.time()
runTime = endTime - startTime
print()
print('Total run time is:',runTime/3600,'hours')
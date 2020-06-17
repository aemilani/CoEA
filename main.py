import os
import datetime
import pickle
import coea
import train
import numpy as np
import matplotlib.pyplot as plt
import dataset as ds
from utils import setup_logger


date = str(datetime.date.today())
time = str(datetime.datetime.now().time())[:8].replace(':', '-')
dirr = 'results/{}/{}'.format(date, time)

os.makedirs(dirr + '/saves')
os.mkdir(dirr + '/logs')

logger_main = setup_logger(logger_name='main', log_path=dirr + '/logs')
logger_ae = setup_logger(logger_name='AE', log_path=dirr + '/logs')
logger_classifier = setup_logger(logger_name='Classifier', log_path=dirr + '/logs')

layer_weights = (-1, -1)  # avg, min
net_weights = (1, -1)  # Rho_MK, ValLoss

n_gens = 3  # Number of generations

logger_main.info('Loading the dataset...')
train_files, test_files, min_value, max_value, train_elbow_points, mission_times = ds.aramis_dataset()
logger_main.info('Dataset loaded successfully.')

# preparing the CoEA data
np.random.seed(0)
coea_train_idx = np.random.randint(0, len(train_files))
# The last 200 timestamps should include the elbow point
while (mission_times[coea_train_idx] - train_elbow_points[coea_train_idx]) >= 200:
    coea_train_idx = np.random.randint(0, len(train_files))
coea_eval_idx = np.random.randint(0, len(train_files))
while (coea_train_idx == coea_eval_idx) or ((mission_times[coea_eval_idx] - train_elbow_points[coea_eval_idx]) >= 200):
    coea_eval_idx = np.random.randint(0, len(train_files))

logger_main.debug('The file used for CoEA training: {}'.format(train_files[coea_train_idx]))
logger_main.debug('The file used for CoEA evaluation: {}'.format(train_files[coea_eval_idx]))

data_train = np.genfromtxt(train_files[coea_train_idx], dtype=np.float32, delimiter=',')[:, :-1]
data_eval = np.genfromtxt(train_files[coea_eval_idx], dtype=np.float32, delimiter=',')[:, :-1]
data_train = ds.normalize(data_train, min_value, max_value)
data_eval = ds.normalize(data_eval, min_value, max_value)

coea_start_time = datetime.datetime.now()

ca = coea.CoEA(pop_size_bits=2,
               n_layer_species=4,
               layer_weights=layer_weights,
               net_weights=net_weights,
               iters=1000,
               net_pop_size=6,
               data_train=data_train,
               data_eval=data_eval)

logger_main.info('The CoEA initialized with network population size of {}, layer population size of {}, and using '
                 '{} iterations for training the AE.'.format(ca.net_pop_size, (2 ** ca.pop_size_bits), ca.iters))

toolbox = ca.toolbox
net_population = ca.net_population
layer_population = ca.layer_population

logger_main.info('Evaluating initial network population...')

# evaluate initial network population
fits = toolbox.map(toolbox.evaluateNet, net_population)
for fit, ind in zip(fits, net_population):
    ind.fitness.values = fit
# order networks population
net_population = ca.order_net_pop(net_population)
ca.layers_credit_assignment(net_population)
# order layers population
for i in range(len(layer_population)):
    layer_population[i] = toolbox.selectNSGA2(layer_population[i], k=ca.layer_pop_size)
logger_main.info('Initial Population Fitness Values:')
for ind in net_population:
    logger_main.info(str(ind.fitness.values))

# for saving populations and fitnesses
avg_layer_fits, avg_net_rho, min_layer_fits, max_net_rho = [], [], [], []
net_pops, layer_pops = [], []

initial_net_pop = toolbox.clone(net_population)
initial_layer_pop = toolbox.clone(layer_population)
for i in range(ca.n_layer_species):
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
    logger_main.info('CoEA generation {} starting...'.format(gen + 1))
    # network population evolution
    # structural and parametric mutation of diverged networks
    for ind in net_population:
        if ind.fitness.values[1] == 100.0:
            toolbox.mutateNetStructure(ind)
            toolbox.mutateNetParameters(ind)
            del ind.fitness.values
    n_net_top_inds = int(0.7 * ca.net_pop_size)  # number of top 70% network individuals
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
    n_layer_top_inds = int(0.7 * ca.layer_pop_size)  # number of top 70% layer individuals
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
                                         size=ca.layer_pop_size - n_layer_top_inds,
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
        for i in range(ca.n_layer_species):
            if ind.net_params['h{}'.format(i + 1)] in deleted_indexes[i]:
                del ind.fitness.values
    invalid_net_inds = [ind for ind in net_population if not ind.fitness.valid]
    fits = map(toolbox.evaluateNet, invalid_net_inds)
    for ind, fit in zip(invalid_net_inds, fits):
        ind.fitness.values = fit
    net_population = ca.order_net_pop(net_population)
    ca.layers_credit_assignment(net_population)
    for i in range(len(layer_population)):
        layer_population[i] = toolbox.selectNSGA2(layer_population[i], k=ca.layer_pop_size)
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
    for i in range(ca.n_layer_species):
        pop[i].sort(key=lambda x: x.index)
    layer_pops.append(pop)
    pop = toolbox.clone(net_population)
    net_pops.append(pop)
    del pop
    logger_main.info('CoEA generation {} ended. Fitness values:'.format(gen + 1))
    for ind in net_population:
        logger_main.info(str(ind.fitness.values))
    elapsed_time = datetime.datetime.now() - coea_start_time
    logger_main.info('Elapsed time: {}'.format(elapsed_time))


coea_run_time = datetime.datetime.now() - coea_start_time
logger_main.info('CoEA ended. Running duration: {}'.format(coea_run_time))

plt.figure()
plt.plot(max_net_rho)
plt.xlabel('Generations')
plt.ylabel('Rho_MK')
plt.title('Max Rho_MK per generation')
plt.savefig(dirr + '/Max_Rho_MK_per_generation.png')

plt.figure()
plt.plot(avg_net_rho)
plt.xlabel('Generations')
plt.ylabel('Rho_MK')
plt.title('Average Rho_MK per generation')
plt.savefig(dirr + '/Average_Rho_MK_per_generation.png')

for i in range(ca.n_layer_species):
    plt.figure()
    plt.plot(np.array(avg_layer_fits)[:, i])
    plt.xlabel('Generation')
    plt.ylabel('"Min" Fitness')
    plt.title('Average "Min" Fitness of the Layer Species nr. {} Per Generation'.format(i))
    plt.savefig(dirr + '/' + 'Average_Min_Fitness_of_the_Layer_Species_nr_{}_Per_Generation.png'.format(i))


final_pop = net_pops[-1]

for i, ae in enumerate(final_pop):
    pickle.dump(ae, open(dirr + '/saves/ae_{}.p'.format(i), 'wb'))


first_front = toolbox.selectNSGA2fronts(final_pop, k=ca.net_pop_size)[0]


# Classification
np.random.seed(7)
np.random.shuffle(train_files)
valid_split = 0.25
valid_files = train_files[:int(valid_split * len(train_files))]
train_files = train_files[int(valid_split * len(train_files)):]

ind = first_front[0]

logger_main.info('Starting the AE training...')
ae_start_time = datetime.datetime.now()
ae, ae_training_history = train.train_ae(ind.net_params, ind.layer_params, train_files, valid_files,
                                         min_value, max_value, n_layers=4, weights=ind.final_weights,
                                         logger=logger_ae)
ae_run_time = datetime.datetime.now() - ae_start_time
logger_main.info('Training the AE ended. Duration: {}'.format(ae_run_time))

ae.save(dirr + '/saves/trained_ae.h5')

plt.figure()
plt.plot(ae_training_history['loss'], label='Training Loss')
plt.plot(ae_training_history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('AE Training')
plt.savefig(dirr + '/AE_training.png')

logger_main.info('Starting the classifier training...')
classification_start_time = datetime.datetime.now()
classifier, classifier_training_history = train.classify(ae, train_files, valid_files, min_value, max_value,
                                                         logger=logger_classifier)
classification_run_time = datetime.datetime.now() - classification_start_time
logger_main.info('Training the classifier ended. Duration: {}'.format(classification_run_time))

classifier.save(dirr + '/saves/trained_classifier.h5')

total_run_time = coea_run_time + ae_run_time + classification_run_time
logger_main.info('Total run time: {}'.format(total_run_time))

plt.figure()
plt.plot(classifier_training_history['loss'], label='Training Loss')
plt.plot(classifier_training_history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Classifier Training - Loss')
plt.savefig(dirr + '/Classifier_training_loss.png')

plt.figure()
plt.plot(classifier_training_history['accuracy'], label='Training Accuracy')
plt.plot(classifier_training_history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Classifier Training - Accuracy')
plt.savefig(dirr + '/Classifier_training_accuracy.png')

plt.figure()
plt.plot(classifier_training_history['aramis_metric'], label='Training Average Error')
plt.plot(classifier_training_history['val_aramis_metric'], label='Validation Average Error')
plt.xlabel('Epochs')
plt.ylabel('Average Error')
plt.legend()
plt.title('Classifier Training - Aramis Metric')
plt.savefig(dirr + '/Classifier_training_metric.png')
plt.show()

# predicted test labels

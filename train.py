import os
import numpy as np
import random as rn
import tensorflow as tf
import dataset as ds
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential, optimizers, layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from utils import calc_rho, aramis_metric


# Defining KL Divergence activity regularizer class
def kl_divergence(rho, rho_hat):
    return abs(rho * tf.math.log(rho) - rho * tf.math.log(rho_hat) +
               (1 - rho) * tf.math.log(1 - rho) - (1 - rho) * tf.math.log(1 - rho_hat))


class SparseActivityRegularizer(regularizers.Regularizer):
    def __init__(self, p=0.05, sparsity_beta=1):
        self.p = K.cast(p, 'float32')
        self.sparsityBeta = K.cast(sparsity_beta, 'float32')

    def __call__(self, x):
        regularization = 0
        p_hat = K.mean(K.abs(x), axis=0)
        regularization += self.sparsityBeta * K.sum(kl_divergence(self.p, p_hat))
        return regularization

    def get_config(self):
        return {"name": self.__class__.__name__}


def train_ae_coea(net_params, layer_params_list, data_train, data_eval, n_layers=4,
                  iters=4000, ind=None):
    """Trains the Autoencoder with the given parameters. Returns max Rho_MK and validation loss"""

    K.clear_session()

    # Required in order to have reproducible results from a specific random seed
    os.environ['PYTHONHASHSEED'] = '0'

    # Force tf to use a single thread (required for reproducibility)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                            inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),
                                config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

    # network parameters
    batch = 32
    optim = net_params['optim']
    learn_rate = net_params['learn_rate']
    decay = net_params['decay']
    mom = net_params['mom']
    rand_seed = net_params['rand_seed']

    np.random.seed(rand_seed)
    rn.seed(rand_seed)
    tf.compat.v1.set_random_seed(rand_seed)

    iter_per_epoch = int(np.ceil(len(data_train) / batch))
    epochs = int(np.ceil(iters / iter_per_epoch))

    if optim == 'adam':
        opt = optimizers.Adam(lr=learn_rate, beta_1=mom, decay=decay, clipvalue=0.3)
    elif optim == 'nadam':
        opt = optimizers.Nadam(lr=learn_rate, beta_1=mom, schedule_decay=decay, clipvalue=0.3)
    elif optim == 'rmsprop':
        opt = optimizers.RMSprop(lr=learn_rate, rho=mom, decay=decay, clipvalue=0.3)
    else:  # adadelta
        opt = optimizers.Adadelta(lr=learn_rate, rho=mom, decay=decay, clipvalue=0.3)

    # pretraining
    input_data = data_train
    weights = []
    ea = EarlyStopping(patience=int(epochs / 3))
    cb = [ea]
    for i, layer_params in enumerate(layer_params_list):
        layer_params = layer_params_list[n_layers - 1 - i]
        input_layer = layers.Input(shape=(input_data.shape[1],))
        hidden_layer = layers.Dense(layer_params['n_neuron'],
                                    activation=layer_params['act'],
                                    kernel_regularizer=regularizers.l2(layer_params['L2']),
                                    activity_regularizer=SparseActivityRegularizer
                                    (p=layer_params['SP'], sparsity_beta=layer_params['SR']),
                                    kernel_initializer=layer_params['init'])(input_layer)
        output_layer = layers.Dense(input_data.shape[1], activation=layer_params['act'],
                                    kernel_initializer=layer_params['init'])(hidden_layer)
        model = models.Model(input_layer, output_layer)
        model.compile(optimizer=opt, loss='mse')
        history = model.fit(x=input_data, y=input_data, batch_size=batch,
                            epochs=epochs,
                            callbacks=cb, validation_split=0.2,
                            verbose=False)
        for loss in history.history['loss']:
            if np.isnan(loss):
                K.clear_session()
                return 0.01, 100
        h_weights = model.get_weights()
        weights.insert(2 * i, h_weights[0])
        weights.insert(2 * i + 1, h_weights[1])
        weights.insert(len(weights) - 2 * i, h_weights[2])
        weights.insert(len(weights) - 2 * i, h_weights[3])
        model = models.Model(input_layer, hidden_layer)
        input_data = model.predict(input_data)

    # stacking the layers - fine tuning
    input_layer = layers.Input(shape=(data_train.shape[1],))
    enc = layers.Dense(layer_params_list[-1]['n_neuron'],
                       activation=layer_params_list[-1]['act'],
                       kernel_initializer=layer_params_list[-1]['init'])(input_layer)
    for i in range(n_layers - 1):
        enc = layers.Dense(layer_params_list[-2 - i]['n_neuron'],
                           activation=layer_params_list[-2 - i]['act'],
                           kernel_initializer=layer_params_list[-2 - i]['init'])(enc)
    dec = layers.Dense(layer_params_list[1]['n_neuron'],
                       activation=layer_params_list[1]['act'],
                       kernel_initializer=layer_params_list[1]['init'])(enc)
    for i in range(n_layers - 2):
        dec = layers.Dense(layer_params_list[i + 2]['n_neuron'],
                           activation=layer_params_list[i + 2]['act'],
                           kernel_initializer=layer_params_list[i + 2]['init'])(dec)
    output_layer = layers.Dense(len(data_train.T),
                                activation=layer_params_list[-1]['act'],
                                kernel_initializer=layer_params_list[-1]['init'])(dec)
    # assumption: output layer has the same parameters as the final hidden layer
    model = models.Model(input_layer, output_layer)
    model.compile(optimizer=opt, loss='mse')
    model.set_weights(weights)
    history = model.fit(x=data_train, y=data_train, batch_size=batch,
                        epochs=epochs,
                        callbacks=cb, validation_data=(data_eval, data_eval),
                        verbose=False)
    if ind:
        ind.final_weights = model.get_weights()
    for loss in history.history['loss']:
        if np.isnan(loss):
            K.clear_session()
            return 0.01, 100
    val_loss = history.history['val_loss'][-1]
    model = models.Model(input_layer, enc)
    indicators = model.predict(data_eval)

    # MK test
    rho_mk = calc_rho(indicators)

    max_rho_mk = 0.01 if np.isnan(max(abs(rho_mk))) else max(abs(rho_mk))
    loss = 100 if np.isnan(val_loss) else val_loss

    return max_rho_mk, loss


def get_monotonic_indicator(net_params, layer_params_list, weights,
                            data, n_layers=4):
    input_layer = layers.Input(shape=(data.shape[1],))
    enc = layers.Dense(layer_params_list[-1]['n_neuron'],
                       activation=layer_params_list[-1]['act'],
                       kernel_initializer=layer_params_list[-1]['init'])(input_layer)
    for i in range(n_layers - 1):
        enc = layers.Dense(layer_params_list[-2 - i]['n_neuron'],
                           activation=layer_params_list[-2 - i]['act'],
                           kernel_initializer=layer_params_list[-2 - i]['init'])(enc)
    dec = layers.Dense(layer_params_list[1]['n_neuron'],
                       activation=layer_params_list[1]['act'],
                       kernel_initializer=layer_params_list[1]['init'])(enc)
    for i in range(n_layers - 2):
        dec = layers.Dense(layer_params_list[i + 2]['n_neuron'],
                           activation=layer_params_list[i + 2]['act'],
                           kernel_initializer=layer_params_list[i + 2]['init'])(dec)
    output_layer = layers.Dense(len(data.T),
                                activation=layer_params_list[-1]['act'],
                                kernel_initializer=layer_params_list[-1]['init'])(dec)
    model = models.Model(input_layer, output_layer)
    model.set_weights(weights)
    model = models.Model(input_layer, enc)
    indicators = model.predict(data)
    rho_mk = np.abs(calc_rho(indicators))
    return indicators[:, np.argmax(rho_mk)]


def train_ae(net_params, layer_params_list, train_files, valid_files, min_value, max_value,
             logger, n_layers=4, weights=None):
    """Trains an AE with the given parameters. Returns the trained AE."""

    K.clear_session()

    # Required in order to have reproducible results from a specific random seed
    os.environ['PYTHONHASHSEED'] = '0'

    # Force tf to use a single thread (required for reproducibility)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                            inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),
                                config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

    # network parameters
    optim = net_params['optim']
    learn_rate = net_params['learn_rate']
    decay = net_params['decay']
    mom = net_params['mom']
    rand_seed = net_params['rand_seed']

    np.random.seed(rand_seed)
    rn.seed(rand_seed)
    tf.compat.v1.set_random_seed(rand_seed)

    if optim == 'adam':
        opt = optimizers.Adam(lr=learn_rate, beta_1=mom, decay=decay, clipvalue=0.3)
    elif optim == 'nadam':
        opt = optimizers.Nadam(lr=learn_rate, beta_1=mom, schedule_decay=decay, clipvalue=0.3)
    elif optim == 'rmsprop':
        opt = optimizers.RMSprop(lr=learn_rate, rho=mom, decay=decay, clipvalue=0.3)
    else:  # adadelta
        opt = optimizers.Adadelta(lr=learn_rate, rho=mom, decay=decay, clipvalue=0.3)

    logger.info('Starting model definition...')

    input_layer = layers.Input(shape=(10,))
    enc = layers.Dense(layer_params_list[-1]['n_neuron'],
                       activation=layer_params_list[-1]['act'],
                       kernel_initializer=layer_params_list[-1]['init'])(input_layer)
    for i in range(n_layers - 1):
        enc = layers.Dense(layer_params_list[-2 - i]['n_neuron'],
                           activation=layer_params_list[-2 - i]['act'],
                           kernel_initializer=layer_params_list[-2 - i]['init'])(enc)
    dec = layers.Dense(layer_params_list[1]['n_neuron'],
                       activation=layer_params_list[1]['act'],
                       kernel_initializer=layer_params_list[1]['init'])(enc)
    for i in range(n_layers - 2):
        dec = layers.Dense(layer_params_list[i + 2]['n_neuron'],
                           activation=layer_params_list[i + 2]['act'],
                           kernel_initializer=layer_params_list[i + 2]['init'])(dec)
    output_layer = layers.Dense(10,
                                activation=layer_params_list[-1]['act'],
                                kernel_initializer=layer_params_list[-1]['init'])(dec)
    # assumption: output layer has the same parameters as the final hidden layer
    ae = models.Model(input_layer, output_layer)
    ae.compile(optimizer=opt, loss='mse')
    if weights:
        ae.set_weights(weights)

    logger.info('Model successfully compiled.')
    logger.info('Network parameters: {}'.format(str(net_params)))
    for i, layer_params in enumerate(layer_params_list):
        logger.info('Layer {} parameters: {}'.format(str(i + 1), str(layer_params)))

    training_history = {}

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    early_stop_counter = 0

    for epoch in range(1, 201):
        logger.info('AE epoch {} starting...'.format(epoch))
        if epoch % 10 == 0:
            ae.save('checkpoints/ae_checkpoint.h5')
            K.clear_session()
            ae = models.load_model('checkpoints/ae_checkpoint.h5')
        for train_file in train_files:
            valid_file = np.random.choice(valid_files)
            logger.debug('Training file: {}'.format(train_file))
            logger.debug('Validation file: {}'.format(valid_file))
            data_train = np.genfromtxt(train_file, dtype=np.float32, delimiter=',')
            data_valid = np.genfromtxt(valid_file, dtype=np.float32, delimiter=',')
            features_train = data_train[:, :-1]
            features_valid = data_valid[:, :-1]
            features_train = ds.normalize(features_train, min_value, max_value)
            features_valid = ds.normalize(features_valid, min_value, max_value)
            history = ae.fit(x=features_train, y=features_train, batch_size=32, epochs=1,
                             validation_data=(features_valid, features_valid), verbose=2)
            for key in history.history:
                if key in training_history.keys():
                    training_history[key].extend(history.history[key])
                else:
                    training_history[key] = history.history[key]
        for key in training_history:
            avg = np.average(training_history[key][-len(train_files):])
            del training_history[key][-len(train_files):]
            training_history[key].append(avg)
        if epoch > 20:
            if not training_history['val_loss'][-1] < training_history['val_loss'][-2]:
                early_stop_counter += 1
                logger.debug('Early stopping counter increased by 1.')
            else:
                if early_stop_counter > 0:
                    early_stop_counter = 0
                    logger.debug('Early stopping counter reset to 0.')
        if (early_stop_counter > 10) or (training_history['val_loss'][-1] < 1e-5):
            logger.info('Training terminated by early stopping.')
            break

    return ae, training_history


def classify(ae, train_files, valid_files, min_value, max_value, logger):

    K.clear_session()

    # Required in order to have reproducible results from a specific random seed
    os.environ['PYTHONHASHSEED'] = '0'

    # Force tf to use a single thread (required for reproducibility)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                            inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),
                                config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

    np.random.seed(13)
    rn.seed(13)
    tf.compat.v1.set_random_seed(13)

    logger.info('Starting model definition...')

    classifier = Sequential()
    for i, layer in enumerate(ae.layers[:int(len(ae.layers) / 2) + 1]):
        classifier.add(layer)
    for layer in classifier.layers:
        layer.trainable = False
    classifier.add(layers.Dense(2, 'sigmoid', name='dense_cls'))
    
    metrics_list = ['accuracy']
    
    classifier.compile('nadam', 'binary_crossentropy', metrics=metrics_list)

    logger.info('Model successfully compiled.')

    training_history = {'aramis_metric': [],
                        'val_aramis_metric': []}

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    early_stop_counter = 0
    best_metric = 1
    best_metric_epoch = 1

    for epoch in range(1, 1001):
        logger.info('Classifier epoch {} starting...'.format(epoch))
        true_labels_train, predicted_labels_train = [], []
        true_labels_valid, predicted_labels_valid = [], []
        if epoch % 10 == 0:
            classifier.save(
                'checkpoints/classifier_checkpoint_epoch_{}.h5'.format(epoch))
            K.clear_session()
            classifier = models.load_model(
                'checkpoints/classifier_checkpoint_epoch_{}.h5'.format(epoch))
        if epoch == 50:
            classifier.layers[-2].trainable = True
        if epoch == 300:
            classifier.layers[-3].trainable = True
        for train_file in train_files:
            valid_file = np.random.choice(valid_files)
            logger.debug('Training file: {}'.format(train_file))
            logger.debug('Validation file: {}'.format(valid_file))
            data_train = np.genfromtxt(train_file, dtype=np.float32, delimiter=',')
            # data_train = ds.balanced_sample(data_train)
            data_train = ds.oversample_smote(data_train)
            data_valid = np.genfromtxt(valid_file, dtype=np.float32, delimiter=',')
            features_train = data_train[:, :-1]
            features_valid = data_valid[:, :-1]
            features_train = ds.normalize(features_train, min_value, max_value)
            features_valid = ds.normalize(features_valid, min_value, max_value)
            labels_train = data_train[:, -1]
            labels_train = to_categorical(labels_train, num_classes=2)
            labels_valid = data_valid[:, -1]
            labels_valid = to_categorical(labels_valid, num_classes=2)

            history = classifier.fit(x=features_train, y=labels_train, batch_size=32, epochs=1,
                                     validation_data=(features_valid, labels_valid), verbose=2)
            true_labels_train.append(labels_train)
            predicted_labels_train.append(classifier.predict(features_train))
            true_labels_valid.append(labels_valid)
            predicted_labels_valid.append(classifier.predict(features_valid))
            for key in history.history:
                if key in training_history.keys():
                    training_history[key].extend(history.history[key])
                else:
                    training_history[key] = history.history[key]
        for i in range(len(true_labels_train)):
            true_labels_train[i] = np.argmax(true_labels_train[i], axis=1)
            predicted_labels_train[i] = np.argmax(predicted_labels_train[i], axis=1)
            true_labels_valid[i] = np.argmax(true_labels_valid[i], axis=1)
            predicted_labels_valid[i] = np.argmax(predicted_labels_valid[i], axis=1)
        metric = aramis_metric(true_labels_train, predicted_labels_train)
        val_metric = aramis_metric(true_labels_valid, predicted_labels_valid)
        training_history['aramis_metric'].append(metric)
        training_history['val_aramis_metric'].append(val_metric)
        logger.info('aramis_metric = {}'.format(metric))
        logger.info('val_aramis_metric = {}'.format(val_metric))
        if (val_metric < best_metric) and (epoch > 600):
            classifier.save('checkpoints/classifier_best.h5')
            best_metric = val_metric
            best_metric_epoch = epoch
        for key in history.history:
            avg = np.average(training_history[key][-len(train_files):])  # average of each epoch
            del training_history[key][-len(train_files):]
            training_history[key].append(avg)
        if epoch > 20:
            if not training_history['val_loss'][-1] < training_history['val_loss'][-2]:
                early_stop_counter += 1
                logger.debug('Early stopping counter increased by 1.')
            else:
                if early_stop_counter > 0:
                    early_stop_counter = 0
                    logger.debug('Early stopping counter reset to 0.')
        if early_stop_counter > 10:
            logger.info('Training terminated by early stopping.')
            break
        
    classifier = models.load_model('checkpoints/classifier_best.h5')

    os.rename(r'checkpoints/classifier_best.h5',
              r'checkpoints/classifier_best_epoch_{}_metric_{}.h5'.
              format(best_metric_epoch, str(best_metric)[2:6]))

    return classifier, training_history


def predict(classifier, test_files, min_value, max_value):
    """Returns a list of arrays of predicted labels."""
    predictions = []
    for test_file in test_files:
        features = np.genfromtxt(test_file, dtype=np.float32, delimiter=',')[:, :10]
        features = ds.normalize(features, min_value, max_value)
        prediction = classifier.predict(features)
        prediction = np.argmax(prediction, axis=1)
        predictions.append(prediction)
    return predictions

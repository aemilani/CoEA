import os
import numpy as np
import random as rn
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models, layers, regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


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


def auto_encoder(net_params, layer_params_list, data_train, data_eval, n_layers=4,
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
    elif optim == 'adadelta':
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

    # stacking the layers - Fine tuning
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


def calc_rho(indicators):
    v = np.zeros(len(indicators.T))
    rho_mk = np.zeros(len(indicators.T))
    for i in range(len(indicators.T)):
        for k in range(len(indicators) - 1):
            for j in range(k + 1, len(indicators)):
                v[i] += np.sign(indicators[j][i] - indicators[k][i])
    n = len(indicators)
    sigma = np.sqrt(n * (n - 1) * (2 * n + 5) / 18)
    for i in range(len(indicators.T)):
        if v[i] > 1:
            rho_mk[i] = (v[i] - 1) / sigma
        elif v[i] == 0:
            rho_mk[i] = 0
        else:
            rho_mk[i] = (v[i] + 1) / sigma
    return rho_mk


def train_ae(net_params, layer_params_list, train_data, valid_data,
             n_layers=4, ind=None):
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
    elif optim == 'adadelta':
        opt = optimizers.Adadelta(lr=learn_rate, rho=mom, decay=decay, clipvalue=0.3)

    ea = EarlyStopping(patience=5)
    cp = ModelCheckpoint('checkpoint.hdf5', save_best_only=True)
    cb = [ea, cp]

    # stacking the layers - Fine tuning
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
    if ind:
        ae.set_weights(ind.final_weights)
    history = ae.fit_generator(train_data, epochs=100,
                               callbacks=cb, validation_data=valid_data)
    return ae, history

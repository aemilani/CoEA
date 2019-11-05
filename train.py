import numpy as np
import random as rn
import tensorflow as tf
from keras import models, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping
from keras import backend as K
import os

# preparing the data
data_norm = np.genfromtxt('data_norm_200.csv', delimiter=',', dtype=np.float32).T

# Defining KL Divergence activity regularizer class
def kl_divergence(rho, rho_hat):
    return abs(rho * tf.log(rho) - rho * tf.log(rho_hat) + \
                (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat))

class SparseActivityRegularizer(regularizers.Regularizer):
    def __init__(self, p=0.05, sparsityBeta=1):
        self.p = K.cast(p, 'float32')
        self.sparsityBeta = K.cast(sparsityBeta, 'float32')
    def __call__(self, x):
        regularization = 0            
        p_hat = K.mean(K.abs(x), axis=0)
        regularization += self.sparsityBeta * K.sum(kl_divergence(self.p, p_hat))
        return regularization
    def get_config(self):
        return {"name": self.__class__.__name__}

def autoEncoder(netParams, layerParamsList, nLayers=4, iters=4000):
    '''Trains the Autoencoder with the given parameters'''
    
    K.clear_session()
    
    # Required in order to have reproducible results from a specific random seed
    os.environ['PYTHONHASHSEED'] = '0'
    
    # Force tf to use a single thread (required for reproducibility)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    
    # network parameters
    batch = netParams['batch']
    optim = netParams['optim']
    learn_rate = netParams['learn_rate']
    decay = netParams['decay']
    mom = netParams['mom']
    rand_seed = netParams['rand_seed']

    np.random.seed(rand_seed)
    rn.seed(rand_seed)
    tf.set_random_seed(rand_seed)
        
    iter_per_epoch = int(np.ceil(len(data_norm)/batch))
    epochs = int(np.ceil(iters/iter_per_epoch))
    
    if optim == 'adam':
        opt = optimizers.Adam(lr=learn_rate, beta_1=mom, decay=decay, clipvalue=0.3)
    elif optim == 'nadam':
        opt = optimizers.Nadam(lr=learn_rate, beta_1=mom, schedule_decay=decay, clipvalue=0.3)
    elif optim =='rmsprop':
        opt = optimizers.RMSprop(lr=learn_rate, rho=mom, decay=decay, clipvalue=0.3)
    elif optim == 'adadelta':
        opt = optimizers.Adadelta(lr=learn_rate, rho=mom, decay=decay, clipvalue=0.3)
    
    # pretraining
    input_data = data_norm
    weights = []
    ea = EarlyStopping(patience=int(epochs/3))
    cb = [ea]
    for i, layerParams in enumerate(layerParamsList):
        layerParams = layerParamsList[nLayers-1-i]
        input_layer = layers.Input(shape=(input_data.shape[1],))
        hidden_layer = layers.Dense(layerParams['n_neuron'],
                                    activation=layerParams['act'],
                                    kernel_regularizer=regularizers.l2(layerParams['L2']),
                                    activity_regularizer=SparseActivityRegularizer\
                                    (p=layerParams['SP'], sparsityBeta=layerParams['SR']),
                                    kernel_initializer=layerParams['init'])(input_layer)
        output_layer = layers.Dense(input_data.shape[1], activation=layerParams['act'],
                                    kernel_initializer=layerParams['init'])(hidden_layer)
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
        weights.insert(2*i, h_weights[0])
        weights.insert(2*i+1, h_weights[1])
        weights.insert(len(weights)-2*i, h_weights[2])
        weights.insert(len(weights)-2*i, h_weights[3])
        model = models.Model(input_layer, hidden_layer)
        input_data = model.predict(input_data)
        
    # stacking the layers - Fine tuning
    input_layer = layers.Input(shape=(data_norm.shape[1],))
    enc = layers.Dense(layerParamsList[-1]['n_neuron'],
                       activation=layerParamsList[-1]['act'],
                       kernel_initializer=layerParamsList[-1]['init'])(input_layer)
    for i in range(nLayers-1):
        enc = layers.Dense(layerParamsList[-2-i]['n_neuron'],
                           activation=layerParamsList[-2-i]['act'],
                           kernel_initializer=layerParamsList[-2-i]['init'])(enc)
    dec = layers.Dense(layerParamsList[1]['n_neuron'],
                       activation=layerParamsList[1]['act'],
                       kernel_initializer=layerParamsList[1]['init'])(enc)
    for i in range(nLayers-2):
        dec = layers.Dense(layerParamsList[i+2]['n_neuron'],
                           activation=layerParamsList[i+2]['act'],
                           kernel_initializer=layerParamsList[i+2]['init'])(dec)
    output_layer = layers.Dense(len(data_norm.T),
                                activation=layerParamsList[-1]['act'],
                                kernel_initializer=layerParamsList[-1]['init'])(dec)
    # assumption: output layer has the same parameters as the final hidden layer
    model = models.Model(input_layer, output_layer)
    model.compile(optimizer=opt, loss='mse')
    model.set_weights(weights)
    history = model.fit(x=data_norm, y=data_norm, batch_size=batch,
                        epochs=epochs,
                        callbacks=cb, validation_split=0.2,
                        verbose=False)
    for loss in history.history['loss']:
        if np.isnan(loss):
            K.clear_session()
            return 0.01, 100
    valLoss = history.history['val_loss'][-1]    
    model = models.Model(input_layer, enc)
    indicators = model.predict(data_norm)
    
    # MK test
    V = np.zeros(len(indicators.T))
    rho_mk = np.zeros(len(indicators.T))
    for i in range(len(indicators.T)):
        for k in range(len(indicators)-1):
            for j in range(k+1, len(indicators)):
                V[i] += np.sign(indicators[j][i] - indicators[k][i])
    N = len(indicators)
    sigma = np.sqrt(N*(N-1)*(2*N+5)/18)
    for i in range(len(indicators.T)):
        if V[i]>1:
            rho_mk[i] = (V[i]-1)/sigma
        elif V[i]==0:
            rho_mk[i] = 0
        else:
            rho_mk[i] = (V[i]+1)/sigma
            
    RHO_MK = 0.01 if np.isnan(max(abs(rho_mk))) else max(abs(rho_mk))
    LOSS = 100 if np.isnan(valLoss) else valLoss
    
    return RHO_MK, LOSS
# update day 3: implemented dropout at layer 2 ( caution: don't keep L = 2 )
# update day 4: implemented minibatch GD along with batchnorm

import numpy as np
import struct
import matplotlib.pyplot as plt

def fix_hyperpara(A0):
    n = [A0.shape[1]]
    L = int(input("Enter the number of layers: "))
    for i in range(1, L+1):
        n.append(int(input("Enter the number of neurons in layer " + str(i) + ": ")))

    learning_rate = float(input("Enter the learning rate: "))
    epochs = int(input("Enter the number of epochs: "))

    return L, n, learning_rate, epochs

# He initialization
def initialise_parameters(L, n):
    parameters = {}
    for i in range(1, L+1):
        parameters["W" + str(i)] = np.random.randn(n[i], n[i-1])*(np.sqrt(2/n[i-1]))
        parameters["b" + str(i)] = np.zeros((n[i], 1))
    return parameters
def initialise_parameters_batchnorm(L, n):
    parameters = {}
    for i in range(1, L+1):
        parameters["W" + str(i)] = np.random.randn(n[i], n[i-1])*(np.sqrt(2/n[i-1]))
        parameters["b" + str(i)] = np.zeros((n[i], 1))
        parameters["gamma" + str(i)] = np.ones((n[i], 1))
        parameters["beta" + str(i)] = np.zeros((n[i], 1))
    return parameters

# mathematical functions
def ReLU(Z):
    return np.maximum(0, Z)
'''
def softmax(Z):
    expZ = np.exp((Z-Z.max(axis=0, keepdims=True))/Z.max(axis=0, keepdims=True)-Z.min(axis=0, keepdims=True)) # min max normalization to avoid overflow
    return expZ / expZ.sum(axis=0, keepdims=True)
'''

def softmax(Z):
    # Subtract the max value in each column for numerical stability
    Z_stable = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z_stable)
    return expZ / expZ.sum(axis=0, keepdims=True)


def compute_cost(activations, y):
    m = y.shape[1]
    y_OHE = np.zeros((10, y.shape[1]))
    for i in range(y.shape[1]):
        y_OHE[y[0,i], i] = 1

    mask = np.int64(activations["A" + str(L)] < 1e-7)
    buff = activations["A" + str(L)] + mask * 1e-7

    cost = -np.sum(y_OHE * np.log(buff)) / m
    return cost

# core algorithms
def forward_prop(activations, parameters, L):

    for i in range(1, L):
        assert isinstance(parameters, dict), "parameters should be a dictionary"
        assert isinstance(activations, dict), "activations should be a dictionary"
        Z = np.dot(parameters["W" + str(i)], activations["A" + str(i-1)]) + parameters["b" + str(i)]
        activations["A" + str(i)] = ReLU(Z)

    Z = np.dot(parameters["W" + str(L)], activations["A" + str(L-1)]) + parameters["b" + str(L)]
    activations["A" + str(L)] = softmax(Z)

    return activations

def back_prop(activations, parameters, y, L):
    grad = {}
    m = y.shape[1]
    y_OHE = np.zeros((10, y.shape[1]))
    for i in range(y.shape[1]):
        y_OHE[y[0,i], i] = 1
    grad["dZ" + str(L)] = activations["A" + str(L)] - y_OHE
    #grad["dA" + str(L)] = -np.divide(y, activations["A" + str(L)])
    grad["dW" + str(L)] = np.dot(grad["dZ" + str(L)], activations["A" + str(L-1)].T) / m
    grad["db" + str(L)] = np.sum(grad["dZ" + str(L)], axis=1, keepdims=True) / m
    grad["dA" + str(L-1)] = np.dot(parameters["W" + str(L)].T, grad["dZ" + str(L)])

    for i in range(L-1, 0, -1):
        grad["dZ" + str(i)] = np.multiply(grad["dA" + str(i)], np.int64(activations["A" + str(i)] > 0))
        grad["dW" + str(i)] = np.dot(grad["dZ" + str(i)], activations["A" + str(i-1)].T) / m
        grad["db" + str(i)] = np.sum(grad["dZ" + str(i)], axis=1, keepdims=True) / m
        grad["dA" + str(i-1)] = np.dot(parameters["W" + str(i)].T, grad["dZ" + str(i)])

    return grad

def update_parameters(parameters, momentum, rms, learning_rate, L):
    for i in range(1, L+1):
        parameters["W" + str(i)] = parameters["W" + str(i)] - learning_rate * momentum["dW" + str(i)]/(np.sqrt(rms["dW" + str(i)]) + 1e-7) # epsilon outside sqrt
        parameters["b" + str(i)] = parameters["b" + str(i)] - learning_rate * momentum["db" + str(i)]/(np.sqrt(rms["db" + str(i)]) + 1e-7)
    return parameters
# core algorithms
def forward_prop_batchnorm(activations, parameters, L):

    cache = {}
    keep_prob = 0.8 # hardcoding here
    for i in range(1, L):
        assert isinstance(parameters, dict), "parameters should be a dictionary"
        assert isinstance(activations, dict), "activations should be a dictionary"
        Z = np.dot(parameters["W" + str(i)], activations["A" + str(i-1)])

        if(i == 2): # dropout only on second layer
            dropout_mask = np.random.rand(Z.shape[0], Z.shape[1])
            dropout_mask = (dropout_mask < keep_prob) / keep_prob # masking and scaling
            Z = np.multiply(Z, dropout_mask)

        Z_norm = (Z - np.mean(Z, axis=1, keepdims=True))/np.sqrt(np.var(Z, axis=1, keepdims=True) + 1e-7)
        cache["var" + str(i)] = np.var(Z, axis = 1, keepdims = True)
        cache["mean" + str(i)] = np.mean(Z, axis = 1, keepdims = True)

        activations["Z" + str(i)] = Z
        activations["Z_norm" + str(i)] = Z_norm
        Z_tilde = parameters["gamma" + str(i)] * Z_norm + parameters["beta" + str(i)]

        activations["A" + str(i)] = ReLU(Z_tilde) # masking Z will automatically mask all terms derived from an elementwise operation on Z


    # care must be taken as we have not considered the case when L = 2 ( don't keep L = 2 )

    Z = np.dot(parameters["W" + str(L)], activations["A" + str(L-1)])
    Z_norm = (Z - np.mean(Z, axis=1, keepdims=True))/np.sqrt(np.var(Z, axis=1, keepdims=True) + 1e-7)
    cache["var" + str(L)] = np.var(Z, axis = 1, keepdims = True)
    cache["mean" + str(L)] = np.mean(Z, axis = 1, keepdims = True)

    Z_tilde = parameters["gamma" + str(L)] * Z_norm + parameters["beta" + str(L)]
    activations["Z" + str(L)] = Z
    activations["Z_norm" + str(L)] = Z_norm

    activations["A" + str(L)] = softmax(Z_tilde)

    return activations, cache, dropout_mask ######## do necessary changes #######

''' equations for batch norm refered from the paper:
Ioffe, Szegedy, 2015, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
'''
def back_prop_batchnorm(activations, parameters, cache, dropout_mask, y, L):##### do nec changes #####

    grad = {}
    m = y.shape[1]
    y_OHE = np.zeros((10, y.shape[1]))
    for i in range(y.shape[1]):
        y_OHE[y[0,i], i] = 1

    grad["dZ_tilde" + str(L)] = activations["A" + str(L)] - y_OHE
    grad["dZ_norm" + str(L)] = grad["dZ_tilde" + str(L)] * parameters["gamma" + str(L)]

    dvar = ( np.sum( grad["dZ_norm" + str(L)] * (activations["Z" + str(L)] - cache["mean" + str(L)]) * ((-1/2)*((cache["var" + str(L)]+1e-7)**(-1.5))), axis = 1, keepdims = True))
    dmean = np.sum(np.multiply(grad["dZ_norm" + str(L)], -1/np.sqrt(cache["var" + str(L)] + 1e-7)),axis = 1, keepdims = True) + dvar * (np.sum(-2*(activations["Z" + str(L)] - cache["mean" + str(L)]))/m)

    grad["dZ" + str(L)] = np.multiply(grad["dZ_norm" + str(L)], 1/np.sqrt(cache["var" + str(L)] + 1e-7)) + dvar * (2*(activations["Z" + str(L)] - cache["mean" +str(L)])/m) + dmean*(1/m)
    #grad["dA" + str(L)] = -np.divide(y, activations["A" + str(L)])
    grad["dgamma" + str(L)] = np.sum(grad["dZ_tilde" + str(L)] * activations["Z_norm" + str(L)], axis = 1, keepdims = True)
    grad["dbeta" + str(L)] = np.sum(grad["dZ_tilde" + str(L)], axis = 1, keepdims = True)

    grad["dW" + str(L)] = np.dot(grad["dZ" + str(L)], activations["A" + str(L-1)].T) / m
    # grad["db" + str(L)] = np.sum(grad["dZ" + str(L)], axis=1, keepdims=True) / m
    grad["dA" + str(L-1)] = np.dot(parameters["W" + str(L)].T, grad["dZ" + str(L)])

    for i in range(L-1, 0, -1):
        grad["dZ_tilde" + str(i)] = np.multiply(grad["dA" + str(i)], np.int64(activations["A" + str(i)] > 0))

        grad["dZ_norm" + str(i)] = np.multiply(grad["dZ_tilde" + str(i)], parameters["gamma" + str(i)])
        dvar = ( np.sum( grad["dZ_norm" + str(i)] * (activations["Z" + str(i)] - cache["mean" + str(i)]) * ((-1/2)*((cache["var" + str(i)]+1e-7)**(-1.5))), axis = 1, keepdims = True))
        dmean = np.sum(np.multiply(grad["dZ_norm" + str(i)], -1/np.sqrt(cache["var" + str(i)] + 1e-7)),axis = 1, keepdims = True) + dvar * (np.sum(-2*(activations["Z" + str(i)] - cache["mean" + str(i)]))/m)
        grad["dZ" + str(i)] = np.multiply(grad["dZ_norm" + str(i)], 1/np.sqrt(cache["var" + str(i)] + 1e-7)) + dvar * (2*(activations["Z" + str(i)] - cache["mean" +str(i)])/m) + dmean*(1/m)
        #grad["dA" + str(l)] = -np.divide(y, activations["A" + str(l)])
        grad["dgamma" + str(i)] = np.sum(grad["dZ_tilde" + str(i)] * activations["Z_norm" + str(i)], axis = 1, keepdims = True)
        grad["dbeta" + str(i)] = np.sum(grad["dZ_tilde" + str(i)], axis = 1, keepdims = True)

        grad["dW" + str(i)] = np.dot(grad["dZ" + str(i)], activations["A" + str(i-1)].T) / m
        # grad["db" + str(i)] = np.sum(grad["dZ" + str(i)], axis=1, keepdims=True) / m
        grad["dA" + str(i-1)] = np.dot(parameters["W" + str(i)].T, grad["dZ" + str(i)])
        if(i-1 == 2):
            grad["dA" + str(i-1)] = grad["dA" + str(i-1)] * dropout_mask


    return grad

def update_parameters_batchnorm(parameters, momentum, grad, rms, learning_rate, L):
    for i in range(1, L+1):
        parameters["W" + str(i)] = parameters["W" + str(i)] - learning_rate * momentum["dW" + str(i)]/(np.sqrt(rms["dW" + str(i)]) + 1e-7) # epsilon outside sqrt
        parameters["gamma" + str(i)] = parameters["gamma" + str(i)] - learning_rate * momentum["dgamma" + str(i)]/(np.sqrt(rms["dgamma" + str(i)] + 1e-7))
        parameters["beta" + str(i)] = parameters["beta" + str(i)] - learning_rate * momentum["dbeta" + str(i)]/(np.sqrt(rms["dbeta" + str(i)] + 1e-7))
        # parameters["b" + str(i)] = parameters["b" + str(i)] - learning_rate * momentum["db" + str(i)]/(np.sqrt(rms["db" + str(i)]) + 1e-7)
    return parameters

def model(x, y, L, n, learning_rate, epochs):
    parameters = initialise_parameters(L, n)

    cost = []
    train_accuracy = []
    activations = {}
    activations["A0"] = x
    m = y.shape[1] # number of training examples
    momentum = {}
    rms = {}
    for i in range(epochs):
        activations = {}
        activations["A0"] = x
        activations = forward_prop(activations, parameters, L)
        cost.append(compute_cost(activations, y))
        grad = back_prop(activations, parameters, y, L)
        for l in range(1, L+1):
            if i == 0:
                momentum["dW" + str(l)] = grad["dW" + str(l)]*0.1
                momentum["db" + str(l)] = grad["db" + str(l)]*0.1
                rms["dW" + str(l)] = 0.001*grad["dW" + str(l)]**2
                rms["db" + str(l)] = 0.001*grad["db" + str(l)]**2
            else:
                momentum["dW" + str(l)] = momentum["dW" + str(l)] * 0.9 + grad["dW" + str(l)]*0.1
                momentum["db" + str(l)] = momentum["db" + str(l)] * 0.9 + grad["db" + str(l)]*0.1
                rms["dW" + str(l)] = rms["dW" + str(l)] * 0.999 + 0.001*grad["dW" + str(l)]**2
                rms["db" + str(l)] = rms["db" + str(l)] * 0.999 + 0.001*grad["db" + str(l)]**2
        parameters = update_parameters(parameters, momentum, rms, learning_rate, L)
        train_accuracy.append(np.sum(np.argmax(activations["A" + str(L)], axis=0) == y) / m)
        if(i+1) % 10 == 0:
            print("Epoch " + str(i+1) + ": " + str(train_accuracy[i]*100) + "%")
            print("Cost: " + str(cost[i]))


    return parameters, cost, train_accuracy

def model_minibatch(x, y, L, n, learning_rate, epochs):
    parameters = initialise_parameters(L, n)

    cost = [] # cost list of all epochs
    train_accuracy = [] # training accuracy list of all epochs
    activations = {}
    activations["A0"] = x
    mb_size = 256 # minibatch size
    num_mb = 60000 // mb_size # number of minibatches
    m = y.shape[1] # number of training examples
    momentum = {}
    rms = {}
    for i in range(epochs):
        cost_batch = [] # cost list of current epoch
        train_accuracy_batch = [] # training accuracy of current epoch
        for j in range(num_mb): # minibatch size = 256, ignoring the 60,000 modulo 256 = 96 examples for simplicity right now
            x_batch = x[:, j*mb_size:(j+1)*mb_size]
            y_batch = y[:, j*mb_size:(j+1)*mb_size]
            activations = {}
            activations["A0"] = x_batch
            activations = forward_prop(activations, parameters, L)
            cost_batch.append(compute_cost(activations, y_batch)) # compute_cost will calculate cost (scalar) of current minibatch
            grad = back_prop(activations, parameters, y_batch, L)
            for l in range(1, L+1):
                if i == 0:
                    momentum["dW" + str(l)] = grad["dW" + str(l)]*0.02
                    momentum["db" + str(l)] = grad["db" + str(l)]*0.02
                    rms["dW" + str(l)] = 0.001*grad["dW" + str(l)]**2
                    rms["db" + str(l)] = 0.001*grad["db" + str(l)]**2
                else:
                    momentum["dW" + str(l)] = momentum["dW" + str(l)] * 0.98 + grad["dW" + str(l)]*0.02
                    momentum["db" + str(l)] = momentum["db" + str(l)] * 0.98 + grad["db" + str(l)]*0.02
                    rms["dW" + str(l)] = rms["dW" + str(l)] * 0.999 + 0.001*grad["dW" + str(l)]**2
                    rms["db" + str(l)] = rms["db" + str(l)] * 0.999 + 0.001*grad["db" + str(l)]**2
            parameters = update_parameters(parameters, momentum, rms, learning_rate, L)
            train_accuracy_batch.append(np.sum(np.argmax(activations["A" + str(L)], axis=0) == y_batch) / mb_size)

        train_accuracy_batch = np.array(train_accuracy_batch)
        train_accuracy.append(np.mean(train_accuracy_batch))
        cost_batch = np.array(cost_batch)
        cost.append(np.mean(cost_batch))
        if(i+1) % 10 == 0:
            print("Epoch " + str(i+1) + ": ")
            print("    Training accuracy: " +  + str(train_accuracy[i]*100) + " %")
            print("    Cost: " + str(cost[i]))


    return parameters, cost, train_accuracy

def model_batchnorm(x, y, L, n, learning_rate, epochs):
    parameters = initialise_parameters_batchnorm(L, n)

    cost = []
    train_accuracy = []
    activations = {}
    activations["A0"] = x
    mb_size = 1024 # minibatch size
    
    m = y.shape[1] # number of training examples
    num_mb = m//1024 # number of minibatches

    momentum = {}
    rms = {}
    batch_cache = {}
    epoch_cache = {}
    
    for i in range(epochs):
        cost_batch = [] # cost list of current epoch
        train_accuracy_batch = [] # training accuracy of current epoch
        for j in range(num_mb +1):
            x_batch = x[:, j*mb_size:(j+1)*mb_size]
            y_batch = y[:, j*mb_size:(j+1)*mb_size]
            activations = {}
            x_batch = (x_batch - np.mean(x_batch))/np.sqrt(np.var(x_batch) + 1e-7)
            activations["A0"] = x_batch
            activations, mb_cache, dropout_mask = forward_prop_batchnorm(activations, parameters, L)
            if j==0:
                for l in range (1, L+1):
                    batch_cache["mean" + str(l)] = 0.1*mb_cache["mean" + str(l)]
                    batch_cache["var" + str(l)] = 0.1*mb_cache["var" + str(l)]
            else: 
                for l in range (1, L+1):
                    batch_cache["mean" + str(l)] = 0.9*batch_cache["mean" + str(l)] + 0.1*mb_cache["mean" + str(l)]
                    batch_cache["var" + str(l)] = 0.9*batch_cache["var" + str(l)] + 0.1*mb_cache["var" + str(l)]

                    
            cost_batch.append(compute_cost(activations, y_batch))
            grad = back_prop_batchnorm(activations, parameters, mb_cache, dropout_mask, y_batch, L)
            for l in range(1, L+1):
                if i == 0:
                    momentum["dW" + str(l)] = grad["dW" + str(l)]*0.1
                    momentum["dgamma" + str(l)] = grad["dgamma" + str(l)]*0.1
                    momentum["dbeta" + str(l)] = grad["dbeta" + str(l)]*0.1
                    #momentum["db" + str(l)] = grad["db" + str(l)]*0.1
                    rms["dW" + str(l)] = 0.001*grad["dW" + str(l)]**2
                    rms["dgamma" + str(l)] = 0.001*grad["dgamma" + str(l)]**2
                    rms["dbeta" + str(l)] = 0.001*grad["dbeta" + str(l)]**2
                    #rms["db" + str(l)] = 0.001*grad["db" + str(l)]**2
                else:
                    momentum["dW" + str(l)] = momentum["dW" + str(l)] * 0.9 + grad["dW" + str(l)]*0.1
                    momentum["dgamma" + str(l)] = momentum["dgamma" + str(l)] * 0.9 + grad["dgamma" + str(l)]*0.1
                    momentum["dbeta" + str(l)] = momentum["dbeta" + str(l)] * 0.9 + grad["dbeta" + str(l)]*0.1
                    #momentum["db" + str(l)] = momentum["db" + str(l)] * 0.9 + grad["db" + str(l)]*0.1
                    rms["dW" + str(l)] = rms["dW" + str(l)] * 0.999 + 0.001*grad["dW" + str(l)]**2
                    rms["dgamma" + str(l)] = rms["dgamma" + str(l)] * 0.999 + 0.001*grad["dgamma" + str(l)]**2
                    rms["dbeta" + str(l)] = rms["dbeta" + str(l)] * 0.999 + 0.001*grad["dbeta" + str(l)]**2
                    #rms["db" + str(l)] = rms["db" + str(l)] * 0.999 + 0.001*grad["db" + str(l)]**2
            parameters = update_parameters_batchnorm(parameters, momentum, grad, rms, learning_rate, L)
            train_accuracy_batch.append(np.sum(np.argmax(activations["A" + str(L)], axis=0) == y_batch) / mb_size)
        
        train_accuracy_batch = np.array(train_accuracy_batch)
        train_accuracy.append(np.mean(train_accuracy_batch))
        cost_batch = np.array(cost_batch)
        cost.append(np.mean(cost_batch))
        if(i+1) % 10 == 0:
            print("Epoch " + str(i+1) + ": ")
            print("Training accuracy: " + str(train_accuracy[i]*100))
            print("Cost: " + str(cost[i]))
            
        if i==0:
            for l in range (1, L+1):
                epoch_cache["mean" + str(l)] = 0.1*batch_cache["mean" + str(l)]
                epoch_cache["var" + str(l)] = 0.1*batch_cache["var" + str(l)]
        else:
            for l in range (1, L+1):
                epoch_cache["mean" + str(l)] = 0.9*epoch_cache["mean" + str(l)] + 0.1*batch_cache["mean" + str(l)]
                epoch_cache["var" + str(l)] = 0.9*epoch_cache["var" + str(l)] + 0.1*batch_cache["var" + str(l)]

            


    return parameters, cost, train_accuracy, epoch_cache

def forward_prop_batchnorm_predict(activations, parameters, epoch_cache, L):

    for i in range(1, L):
        mean = epoch_cache["mean" + str(i)]
        var = epoch_cache["var" + str(i)]
        assert isinstance(parameters, dict), "parameters should be a dictionary"
        assert isinstance(activations, dict), "activations should be a dictionary"
        Z = np.dot(parameters["W" + str(i)], activations["A" + str(i-1)])
        Z_norm = (Z - mean)/np.sqrt(var + 1e-7)

        activations["Z" + str(i)] = Z
        activations["Z_norm" + str(i)] = Z_norm
        Z_tilde = parameters["gamma" + str(i)] * Z_norm + parameters["beta" + str(i)]

        activations["A" + str(i)] = ReLU(Z_tilde)



    Z = np.dot(parameters["W" + str(L)], activations["A" + str(L-1)])
    Z_norm = (Z - np.mean(Z, axis=1, keepdims=True))/np.sqrt(np.var(Z, axis=1, keepdims=True) + 1e-7)

    Z_tilde = parameters["gamma" + str(L)] * Z_norm + parameters["beta" + str(L)]
    activations["Z" + str(L)] = Z
    activations["Z_norm" + str(L)] = Z_norm

    activations["A" + str(L)] = softmax(Z_tilde)

    return activations ######## do necessary changes #######

def predict(x, parameters, epoch_cache, L):
    activations = {}
    activations["A0"] = x
    activations = forward_prop_batchnorm_predict(activations, parameters, epoch_cache, L)
    return np.argmax(activations["A" + str(L)], axis=0)

def accuracy(y, y_pred):
    return np.sum(y == y_pred) / y.shape[1]

# importing dataset
# source: https://stackoverflow.com/questions/62958011/how-to-correctly-parse-mnist-datasetidx-format-into-python-arrays
with open('train-images.idx3-ubyte', 'rb') as f:
    magic, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))
    images = np.fromfile(f, dtype=np.uint8).reshape(num_images, num_rows, num_cols)

with open('train-labels.idx1-ubyte', 'rb') as f:
    magic, num_labels = struct.unpack(">II", f.read(8))
    labels = np.fromfile(f, dtype=np.uint8)

with open('t10k-images.idx3-ubyte', 'rb') as f:
    magic, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))
    images_test = np.fromfile(f, dtype=np.uint8).reshape(num_images, num_rows, num_cols)

with open('t10k-labels.idx1-ubyte', 'rb') as f:
    magic, num_labels = struct.unpack(">II", f.read(8))
    labels_test = np.fromfile(f, dtype=np.uint8)

x_train_flat = images.reshape(images.shape[0], -1) # flattening rows and columns
x_train_flat = x_train_flat.T
x_train_flat = (x_train_flat - np.mean(x_train_flat))/np.sqrt(np.var(x_train_flat) + 1e-7)
# print(x_train_flat.shape)
# x_train_flat.shape = (784, 60000)

y_train = labels.reshape(-1, 1) # converting rank 1 to rank 2 array
y = y_train.T
# print(y.shape)
# y.shape = (1, 60000)

x_test_flat = images_test.reshape(images_test.shape[0], -1) # flattening rows and columns
x_test_flat = x_test_flat.T
x_test_flat = (x_test_flat-np.mean(x_test_flat))/np.sqrt(np.var(x_test_flat) + 1e-7)
# print(x_test_flat.shape)
# x_test_flat.shape = (784, 10000)

y_test = labels_test.reshape(-1, 1) # converting rank 1 to rank 2 array
y_test = y_test.T
# print(y_test.shape)
# y_test.shape = (1, 10000)

# L, n, learning_rate, epochs = fix_hyperpara(x_train_flat)
L = 3
n = [784, 100, 50, 10]
learning_rate = 0.5
epochs = 100
print("No. of hidden layers: ", L-1)
print("array containing no. of units per layer: ", n)
print("learning rate: ", learning_rate)


parameters, cost, train_accuracy, epoch_cache = model_batchnorm(x_train_flat, y, L, n, learning_rate, epochs)

y_pred = predict(x_test_flat, parameters, epoch_cache, L)
print("Test Accuracy: " + str(accuracy(y_test, y_pred)*100) + " %")

plt.plot(range(1, epochs+1), cost, 'r')
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.show()

train_accuracy = np.array(train_accuracy)
train_accuracy = train_accuracy.reshape(epochs, 1)
plt.plot(range(1, epochs+1), train_accuracy*100, 'b')
plt.xlabel("Epochs")
plt.ylabel("Train Accuracy (%)")
plt.show()












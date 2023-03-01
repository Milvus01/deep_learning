from time import time
import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles
from tqdm import tqdm

def initialisation(dimensions):
    parameters = {}
    C = len(dimensions)
    for c in range(1, C):
        parameters['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
        parameters['b' + str(c)] = np.random.randn(dimensions[c], 1)
    return parameters
def forward_propagation(X, parameters):
    activations = {'A0': X}
    C = len(parameters) // 2
    for c in range(1, C + 1):
        Z = parameters['W' + str(c)].dot(activations['A' + str(c - 1)]) + parameters['b' + str(c)]
        activations['A' + str(c)] = 1 / (1 + np.exp(-Z))
    return activations
def back_propagation(X, y, activations, parameters):
    gradients = {}
    m = y.shape[1]
    C = len(parameters) // 2
    dZ = activations['A' + str(C)] - y
    for c in range(C, 0, -1):
        gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T)
        gradients['db' + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dZ = np.dot(parameters['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])
    return gradients
def update(gradients, parameters, learing_rate):
    C = len(parameters) // 2
    for c in range(1, C +1):
        parameters['W' + str(c)] -= learing_rate * gradients['dW' + str(c)]
        parameters['b' + str(c)] -= learing_rate * gradients['db' + str(c)]
    return parameters

def neural_network(X_train, y_train, hidden_layers=(32, 32, 32), learning_rate=0.1, n_iter=1000):
    dimensions = list(hidden_layers)
    dimensions.insert(0, X_train.shape[0])
    dimensions.append(y_train.shape[0])
    parameters = initialisation(dimensions)
    train_loss = [] ; train_acc = []
    for i in tqdm(range(n_iter)):
        activations = forward_propagation(X_train, parameters)
        gradiens = back_propagation(X_train, y_train, activations, parameters)
        parameters = update(gradiens, parameters, learning_rate)
        if i % 100 == 0:
            c = len(parameters) // 2
            train_loss.append(log_loss(y_train, activations['A' + str(c)]))
            # y_pred = predict(X_train, parameters)
            # train_acc.append(accuracy_score(y_train.flatten(), y_pred.flatten()))
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(train_loss, label='train loss') ; plt.legend()
    # plt.subplot(1, 2, 2)
    # plt.plot(train_acc, label='train acc') ; plt.legend()
    # plt.show()
    return parameters

def predict(X, parameters):
    activations = forward_propagation(X, parameters)
    C = len(parameters) // 2
    Af = activations['A' + str(C)]
    return Af >= 0.5

def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels
    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # your train set features
    y_test = np.array(test_dataset["Y_test"][:]) # your train set labels
    return X_train, y_train, X_test, y_test

log_loss = lambda y, activations, epsilon=1e-15 : 1 / len(y) * np.sum(-y * np.log(activations + epsilon) - (1 - y) * np.log(1 - activations + epsilon))

# chdir("/media/maelien/WEBER/prgm/Python/deep_learning")
# (X_train, y_train, X_test, y_test) = load_data()
# X_train = X_train.reshape(X_train.shape[0], -1) / X_train.max()
# X_test = X_test.reshape(X_test.shape[0], -1) / X_train.max()
# parameters = neural_network(X_train, y_train, learning_rate=0.01)
temp1 = time()
X, y = make_circles(n_samples=100, noise=0.1, factor=0.3)
X = X.T
y = y.reshape((1, y.shape[0]))
# plt.scatter(X[0, :], X[1, :], c=y, cmap='summer') ; plt.show()

parameters = neural_network(X, y, hidden_layers=(32, 32, 32), n_iter=10000, learning_rate=0.1)
temp2 = time()
print(temp2 - temp1)

# parameters = { 'W1', 'b1', 'W2', 'b2' }
# activations = { 'A1', 'A2' }
# graidents = { 'dW1', 'db1', 'dW2', 'db2' }

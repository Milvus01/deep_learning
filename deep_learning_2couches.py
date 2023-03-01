from os import chdir

import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles
from tqdm import tqdm

def initialisation(n0, n1, n2):
    return {
        'W1' : np.random.randn(n1, n0),
        'b1' : np.random.randn(n1, 1), # ou np.zeros((n1, 1))
        'W2' : np.random.randn(n2, n1),
        'b2' : np.random.randn(n2, 1), # ou np.zeros((n2, 1))
    } # parameters
def forward_propagation(X, parameters):
    W1 = parameters['W1'] ; b1 = parameters['b1'] ; W2 = parameters['W2'] ; b2 = parameters['b2']
    Z1 = W1.dot(X) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = W2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))
    return {
        'A1' : A1,
        'A2' : A2
    } # activation
def log_loss(A, y, epsilon=1e-15):
    "Calcul du coût"
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))
def back_propagation(X, y, activations, parameters):
    A1 = activations['A1'] ; A2 = activations['A2'] ; W2 = parameters['W2']
    m = y.shape[1]

    dZ2 = A2 - y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims = True)

    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims = True)
    return {
        'dW1' : dW1,
        'db1' : db1,
        'dW2' : dW2,
        'db2' : db2,
    } # gradients
def update(gradients, parameters, learing_rate):
    W1 = parameters['W1'] ; b1 = parameters['b1'] ; W2 = parameters['W2'] ; b2 = parameters['b2']
    dW1 = gradients['dW1'] ; db1 = gradients['db1'] ; dW2 = gradients['dW2'] ; db2 = gradients['db2']
    W1 = W1 - learing_rate * dW1
    b1 = b1 - learing_rate * db1
    W2 = W2 - learing_rate * dW2
    b2 = b2 - learing_rate * db2
    return {
        'W1' : W1,
        'b1' : b1,
        'W2' : W2,
        'b2' : b2,
    } # parameters

def neural_network(X_train, y_train, n1, learning_rate=0.1, n_iter=1000):
    n0 = X_train.shape[0] ; n2 = y_train.shape[0]
    parameters = initialisation(n0, n1, n2)
    train_loss = [] ; train_acc = [] ; history = []
    for i in tqdm(range(n_iter)):
        activations = forward_propagation(X_train, parameters)
        gradiens = back_propagation(X_train, y_train, activations, parameters)
        parameters = update(gradiens, parameters, learning_rate)
        if i % 100 == 0:
            train_loss.append(log_loss(y.flatten(), activations['A2'].flatten()))
            y_pred = predict(X, parameters)
            train_acc.append(accuracy_score(y.flatten(), y_pred.flatten()))
            history.append([parameters.copy(), train_loss, train_acc, i])
    # y_pred = predict(X_train, parameters)
    print(accuracy_score(y_train, y_pred))
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss') ; plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc') ; plt.legend()
    plt.show()
    return parameters

def predict(X, parameters):
    activations = forward_propagation(X, parameters)
    return activations['A2'] >= 0.5

def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels
    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # your train set features
    y_test = np.array(test_dataset["Y_test"][:]) # your train set labels
    return X_train, y_train, X_test, y_test

# chdir("/media/maelien/WEBER/prgm/Python/deep_learning")
# (X_train, y_train, X_test, y_test) = load_data()
# X_train = X_train.reshape(X_train.shape[0], -1) / X_train.max()
# X_test = X_test.reshape(X_test.shape[0], -1) / X_train.max()
# parameters = neural_network(X_train, y_train, learing_rate=0.01)

X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
X = X.T
y = y.reshape((1, y.shape[0]))
plt.scatter(X[0, :], X[1, :], c=y, cmap='summer') ; plt.show()

parameters = neural_network(X, y, n1=8, n_iter=10000, learning_rate=0.1)


# parameters = { 'W1', 'b1', 'W2', 'b2' }
# activations = { 'A1', 'A2' }
# graidents = { 'dW1', 'db1', 'dW2', 'db2' }

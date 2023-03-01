from os import chdir

import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)
def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A
def log_loss(A, y, epsilon=1e-15):
    "Calcul du coût"
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))
def gradiens(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)
def update(dW, db, W, b, learing_rate):
    W = W - learing_rate * dW
    b = b - learing_rate * db
    return (W, b)

def artificial_neuron(X, y, learing_rate=0.1, n_iter=1000):
    W, b = initialisation(X)
    loss = []
    for _ in tqdm(range(n_iter)):
        A = model(X, W, b)
        if _ % 10:
            loss.append(log_loss(A, y))
        dW, db = gradiens(A, X, y)
        W, b = update(dW, db, W, b, learing_rate)
    y_pred = predict(X, W, b)
    print(accuracy_score(y, y_pred))
    plt.plot(loss) ; plt.show()
    return (W, b)
def predict(X, W, b):
    A = model(X, W, b)
    # print(A*100)
    return A >= 0.5
def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels
    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # your train set features
    y_test = np.array(test_dataset["Y_test"][:]) # your train set labels
    return X_train, y_train, X_test, y_test

chdir("/media/maelien/WEBER/prgm/Python/deep_learning")
(X_train, y_train, X_test, y_test) = load_data()
X_train = X_train.reshape(X_train.shape[0], -1) / X_train.max()
X_test = X_test.reshape(X_test.shape[0], -1) / X_train.max()

(W, b) = artificial_neuron(X_train, y_train, learing_rate=0.01)

# X, y = sklearn.datasets.make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
# y = y.reshape((y.shape[0], 1))
# print("dimension de X :", X.shape) ; print("dimension de y :", y.shape)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer') ; plt.show()
# new_plant = np.array([2, 1])
# x0 = np.linspace(-1, 4, 100)
# x1 = (-W[0] * x0 - b) / W[1]
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
# plt.scatter(new_plant[0], new_plant[1], c='r')
# plt.plot(x0, x1, c='orange', lw=3)
# plt.show()
# predict(new_plant, W, b)

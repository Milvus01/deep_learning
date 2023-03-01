from time import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles
from tqdm import tqdm

# par => parameters
# act => activations
# gra => gradiens
# X => vecteur contenant les donnees
# y => vecteur contenant la solution (donnees de reference)

def initialisation(dim):
    """Initialise un reseau de neuronnes :
        (4, 6, 5)
        -> 3 couches respectivement de 4, 6 et 5 neuronnes"""
    par = {}
    for c in range(1, len(dim)):
        par[f'W{c}'] = np.random.randn(dim[c], dim[c - 1])
        par[f'b{c}'] = np.random.randn(dim[c], 1)
    return par
def forward_propagation(par):
    act ={'A0': X}
    for c in range(1, len(par) // 2 + 1):
        Z = par[f'W{c}'].dot(act[f'A{c - 1}']) + par[f'b{c}']
        act[f'A{c}'] = 1 / (1 + np.exp(-Z)) # fonction sigmoide
    return act
def back_propagation(par, act):
    gra = {}
    m = y.shape[1]
    dZ = act[f'A{len(par) // 2}'] - y
    for c in range(len(par) // 2, 0, -1):
        gra[f'dW{c}'] = 1 / m * np.dot(dZ, act[f'A{c - 1}'].T)
        gra[f'db{c}'] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dZ = np.dot(par[f'W{c}'].T, dZ) * act[f'A{c - 1}'] * (1 - act[f'A{c - 1}'])
    return gra
def update(par, gra, learning_rate):
    for c in range(1, len(par) // 2 + 1):
        par[f'W{c}'] -= learning_rate * gra[f'dW{c}']
        par[f'b{c}'] -= learning_rate * gra[f'db{c}']
    return par

def neural_network(_X, _y, n_iter=1000, learning_rate=0.1, hidden_layer=(32, 32, 32), sample=100):
    global X, y
    X = _X ; y = _y
    dim = list(hidden_layer)
    dim.insert(0, X.shape[0])
    dim.append(y.shape[0])
    par = initialisation(dim)
    loss = acc = []
    for i in tqdm(range(n_iter)):
        act = forward_propagation(par)
        gra = back_propagation(par, act)
        par  = update(par, gra, learning_rate)
        if not i % sample:
            c = len(par) // 2
            loss.append(log_loss(y, act[f'A{c}']))
    # plt.plot(loss) ; plt.show()
    return par

log_loss = lambda y, activations, epsilon=1e-15 : 1 / len(y) * np.sum(-y * np.log(activations + epsilon) - (1 - y) * np.log(1 - activations + epsilon))
a = lambda x : 1 / (1 + 2.718281828459045**-x) # fonction sigmoide
temp1 = time()
X, y = make_circles(n_samples=100, noise=0.1, factor=0.3)
X = X.T
y = y.reshape((1, y.shape[0]))
# plt.scatter(X[0, :], X[1, :], c=y, cmap='summer') ; plt.show()

neural_network(X, y, n_iter=10000)
temp2 = time()
print(temp2 - temp1)
from time import time
import matplotlib as plt
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
def forward_propagation(X, par):
    act ={'A0': X}
    for c in range(1, len(par) // 2 + 1):
        Z = par[f'W{c}'].dot(act[f'A{c - 1}']) + par[f'b{c}']
        act[f'A{c}'] = 1 / (1 + np.exp(-Z)) # fonction sigmoide
    return act
def back_propagation(par, act):
    gra = {}
    dZ = act[f'A{len(par) // 2}'] - y
    return gra
def update(par, gra, learning_rate):
    return par

def neural_netowrk(X, y, n_iter=1000, learning_rate=0.1, hidden_layer=(32, 32, 32)):
    len = list(hidden_layer)
    len.insert(0, X.shape[0])
    len.append(y.shape[0])
    par = initialisation(len)

    return par

a = lambda x : 1 / (1 + 2.718281828459045**-x) # fonction sigmoide


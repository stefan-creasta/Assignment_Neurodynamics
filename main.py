import random
import matplotlib.pyplot as plt
import iznetwork
import numpy as np
from numpy import random

from iznetwork import IzNetwork


def initialize_weights(inhibitory_neurons, excitatory_neurons, nr_populations):
    n = inhibitory_neurons + excitatory_neurons
    W = np.zeros((n, n))
    neurons_per_population = excitatory_neurons / nr_populations
    for population in range(nr_populations):
        for it in range(1000):
            x = int(population * neurons_per_population + random.randint(0, 99))
            y = int(population * neurons_per_population + random.randint(0, 99))
            while W[x][y] != 0 or x == y:
                x = int(population * neurons_per_population + random.randint(0, 99))
                y = int(population * neurons_per_population + random.randint(0, 99))
            #print(it)
            W[x][y] = 17

    for it in range(inhibitory_neurons):
        inhibitory = int(nr_populations * neurons_per_population + it)
        for population in range(nr_populations):
            for i in range(4):
                x = int(population* neurons_per_population + random.randint(0, 99))
                while W[x][inhibitory] != 0:
                    x = int(population * neurons_per_population + random.randint(0, 99))
                W[x][inhibitory] = 50 * random.random()
        for excitatory in range(excitatory_neurons):
            W[inhibitory][excitatory] = -2 * random.random()
        for it2 in range(inhibitory_neurons):
            inhibitory_2 = int(nr_populations * neurons_per_population + it2)
            if inhibitory_2 != inhibitory:
                W[inhibitory][inhibitory_2] = -1 * random.random()

    return W

def rewire(W, excitatory_neurons, inhibitory_neurons, p, nr_populations):
    neurons_per_population = int(excitatory_neurons / nr_populations)
    for population in range(nr_populations):
        for it in range(neurons_per_population):
            for it2 in range(neurons_per_population):
                neuron1 = it + population * neurons_per_population
                neuron2 = it2 + population * neurons_per_population
                if W[neuron1][neuron2] != 0 and p >= random.random():
                    new_population = random.randint(0, nr_populations - 1)
                    while new_population == population:
                        new_population = random.randint(0, nr_populations - 1)
                    new_neuron = random.randint(0, neurons_per_population - 1) + new_population * neurons_per_population
                    while W[neuron1][new_neuron] != 0:
                        new_neuron = random.randint(0, neurons_per_population - 1) + new_population * neurons_per_population
                    W[neuron1][neuron2] = 0
                    W[neuron1][new_neuron] = 17

def plot(inhibitory_neurons, excitatory_neurons, nr_populations):
    W = initialize_weights(inhibitory_neurons, excitatory_neurons, nr_populations)
    rewire(W, excitatory_neurons, inhibitory_neurons, p, nr_populations)
    W_p = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if W[i][j] != 0:
                W_p[i][j] = 1
    plt.figure(figsize=(6, 6))
    plt.imshow(W_p, cmap='binary', interpolation='nearest')
    plt.show()
    return W

def get_delays(W, inhibitory_neurons, excitatory_neurons, D_max):
    n = inhibitory_neurons + excitatory_neurons
    D = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if W[i][j] != 0:
                if i < excitatory_neurons and j < excitatory_neurons:
                    D[i][j] = int(random.randint(1, 20))
                else:
                    D[i][j] = int(1)
            else:
                D[i][j] = D_max + 1
    return D

def background_firing(inhibitory_neurons, excitatory_neurons):
    n = inhibitory_neurons + excitatory_neurons
    x = random.poisson(lam=0.01, size=n)
    I = np.zeros(n)
    for i in range(n):
        if x[i] > 0:
            I[i] = 15
    return I

def simulation(T, Network, inhibitory_neurons, excitatory_neurons):
    n = inhibitory_neurons + excitatory_neurons
    V = np.zeros((T, n))
    for t in range(T):
        Network.setCurrent(background_firing(inhibitory_neurons, excitatory_neurons))
        Network.update()
        V[t, :], _ = Network.getState()
    plt.plot(V[:, 1])
    t, m = np.where(V > 29)
    print(m.shape)
    print(t.shape)
    mat = np.zeros((T, n))
    for i in range(len(t)):
        if m[i] < 800:
            mat[t[i], m[i]] = 1
    x, y = np.nonzero(mat)
    #plt.scatter(x, y)
    plt.show()

inhibitory_neurons = 200
excitatory_neurons = 800
n = inhibitory_neurons + excitatory_neurons
nr_populations = 8
p = 0
D_max = 20
W = plot(inhibitory_neurons, excitatory_neurons, nr_populations)
D = get_delays(W, inhibitory_neurons, excitatory_neurons, D_max)

a = [0.02] * np.ones(1000)
b_ = [0.2] * 800 + [0.25] * 200
b = np.array(b_)
c = np.array([-65] * 1000)
d = np.array([8] * 800 + [2] * 200)

Network = IzNetwork(n, D_max)
Network.setDelays(D)
Network.setWeights(W)
Network.setParameters(a, b, c, d)

T = 1000
simulation(T, Network, inhibitory_neurons, excitatory_neurons)
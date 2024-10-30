import random
import matplotlib.pyplot as plt
import iznetwork
import numpy as np

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

inhibitory_neurons = 200
excitatory_neurons = 800
n = inhibitory_neurons + excitatory_neurons
nr_populations = 8
p = 0.2
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
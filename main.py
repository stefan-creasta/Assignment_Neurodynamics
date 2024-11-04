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

    #excitatory-excitatory
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
        inhibitory = int(excitatory_neurons + it)
        #inhibitory-excitatory
        for excitatory in range(excitatory_neurons):
            W[inhibitory][excitatory] = -2 * random.random()
        #inhibitory-inhibitory
        for it2 in range(inhibitory_neurons):
            inhibitory_2 = int(excitatory_neurons + it2)
            if inhibitory_2 != inhibitory:
                W[inhibitory][inhibitory_2] = -1 * random.random()
    #excitatory-inhibitory
    connected = set()
    excitatory_list = [i for i in range(int(neurons_per_population))]
    inhibitory_list = [excitatory_neurons + i for i in range(inhibitory_neurons)]
    random.shuffle(inhibitory_list)
    #print(inhibitory_list)
    inhibitory_index = 0
    for population in range(nr_populations):
        random.shuffle(excitatory_list)
        #print(excitatory_list)
        for excitatory in range(int(neurons_per_population)):
            W[excitatory_list[excitatory] + population * int(neurons_per_population)][inhibitory_list[inhibitory_index]] = 50 * random.random()
            #print(str(nr_populations * int(neurons_per_population)) + ", " + str(inhibitory_list[inhibitory_index]))
            if excitatory % 4 == 3:
                inhibitory_index += 1
            # W[excitatory + nr_populations * int(neurons_per_population) + 1][inhibitory_list[inhibitory_index]] = 50 * random.random()
            # W[excitatory + nr_populations * int(neurons_per_population) + 2][inhibitory_list[inhibitory_index]] = 50 * random.random()
            # W[excitatory + nr_populations * int(neurons_per_population) + 3][inhibitory_list[inhibitory_index]] = 50 * random.random()
            #inhibitory_index += 1
    # for population in range(nr_populations):
    #     inhibitory = int(nr_populations * neurons_per_population + random.randint(0, 199))
    #     while inhibitory in connected:
    #         inhibitory = int(nr_populations * neurons_per_population + random.randint(0, 199))
    #     connected.add(inhibitory)
    #     for i in range(4):
    #         x = int(population * neurons_per_population + random.randint(0, 99))
    #         if W[x][inhibitory] != 0:
    #             x = int(population * neurons_per_population + random.randint(0, 99))
    #         W[x][inhibitory] = 50 * random.random()
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
                D[i][j] = 1
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
    t, m = np.where(V > 29)
    print(m.shape)
    print(t.shape)
    mat = np.zeros((T, n))
    for i in range(len(t)):
        if m[i] < 800:
            mat[t[i], m[i]] = 1
    x, y = np.nonzero(mat)
    fig, axs = plt.subplots(1, 1, figsize=(16, 4))
    axs.set_xticks(list(range(0,1001,100)))
    axs.set_yticks(list(range(0, 801, 200)))
    axs.invert_yaxis()
    axs.scatter(x, y)
    plt.xlim(left=0)
    plt.show()
    return V


def compute_firing_rate(V, excitatory_neurons, nr_populations, window_size=50, step_size=20):
    neurons_per_population = excitatory_neurons // nr_populations
    total_time = V.shape[0]
    firing_rates = []

    for pop in range(nr_populations):
        module_firings = V[:, pop * neurons_per_population : (pop + 1) * neurons_per_population] > 29
        module_firing_counts = module_firings.sum(axis=1)  # Total firings per timestep for this module

        # Sliding window for downsampling
        module_rates = []
        for start in range(0, total_time - window_size + 1, step_size):
            window_firings = module_firing_counts[start:start + window_size]
            mean_rate = np.mean(window_firings)  # Average firing rate in this window
            module_rates.append(mean_rate)
        firing_rates.append(module_rates)

    return np.array(firing_rates)  # Shape: (nr_populations, number of windows)

def plot_firing_rates(firing_rates, nr_populations, window_size=50, step_size=20):
    time_points = np.arange(0, len(firing_rates[0]) * step_size, step_size)
    plt.figure(figsize=(10, 4))

    for pop in range(nr_populations):
        plt.plot(time_points, firing_rates[pop], label=f'Module {pop + 1}')

    plt.xlabel('Time (ms)')
    plt.ylabel('Mean Firing Rate (per 50ms window)')
    plt.title('Mean Firing Rate in Each Module')
    plt.legend()
    plt.show()

inhibitory_neurons = 200
excitatory_neurons = 800
n = inhibitory_neurons + excitatory_neurons
nr_populations = 8
p = 0.875
D_max = 20
W = plot(inhibitory_neurons, excitatory_neurons, nr_populations)
D = get_delays(W, inhibitory_neurons, excitatory_neurons, D_max)

a = np.zeros(1000)
b = np.zeros(1000)
c = np.zeros(1000)
d = np.zeros(1000)
for i in range(1000):
    a[i] = 0.02
    c[i] = -65
    r = random.rand()
    if i < 800:
        b[i] = 0.2
        d[i] = 8 - 6 * (r * r)
        c[i] += 15 * (r * r)
    else:
        a[i] += 0.08 * r
        b[i] = 0.25 - 0.05 * r


Network = IzNetwork(n, D_max)
Network.setDelays(D)
Network.setWeights(W)
Network.setParameters(a, b, c, d)

T = 1000
V = simulation(T, Network, inhibitory_neurons, excitatory_neurons)
firing_rates = compute_firing_rate(V, excitatory_neurons, nr_populations)
plot_firing_rates(firing_rates, nr_populations)
import matplotlib.pyplot as plt
import numpy as np
from iznetwork import IzNetwork

"""
Initialize network edges with appropriate weights:
Excitatory to Excitatory - 1000 random edges generated for each population with a weight of 17
Excitatory to Inhibitory - in ascending order, 4 excitatory neurons are connected to 
                           one inhibitory neuron (also in ascending order) with a random weight between [0, 50)
Inhibitory to Excitatory - an edge between each inhibitory to each excitatory neuron exists with a random weight between (-2, 0]
Inhibitory to Inhibitory - an edge between each inhibitory to each other inhibitory neuron exists with a random weight between (-1, 0]

Inputs:
inhibitory_neurons - number of inhibitory neurons (200)
excitatory_neurons - number of excitatory neurons (800)
nr_populations - number of populations (8)
"""
def initialize_weights(inhibitory_neurons, excitatory_neurons, nr_populations):
    n = inhibitory_neurons + excitatory_neurons
    W = np.zeros((n, n))
    neurons_per_population = excitatory_neurons / nr_populations

    #excitatory-excitatory
    for population in range(nr_populations):
        for it in range(1000):
            x = int(population * neurons_per_population + np.random.randint(0, 100))
            y = int(population * neurons_per_population + np.random.randint(0, 100))
            while W[x][y] != 0 or x == y:
                x = int(population * neurons_per_population + np.random.randint(0, 100))
                y = int(population * neurons_per_population + np.random.randint(0, 100))
            W[x][y] = 17

    for it in range(inhibitory_neurons):
        inhibitory = int(excitatory_neurons + it)
        #inhibitory-excitatory
        for excitatory in range(excitatory_neurons):
            W[inhibitory][excitatory] = -2 * np.random.random()
        #inhibitory-inhibitory
        for it2 in range(inhibitory_neurons):
            inhibitory_2 = int(excitatory_neurons + it2)
            if inhibitory_2 != inhibitory:
                W[inhibitory][inhibitory_2] = -1 * np.random.random()

    # Excitatory-Inhibitory Connections
    for i in range(inhibitory_neurons):
        inhibitory_index = excitatory_neurons + i
        excitatory_indices = list(range(i * 4, (i + 1) * 4))
        for excitatory_index in excitatory_indices:
            if excitatory_index < excitatory_neurons:
                W[excitatory_index][inhibitory_index] = 50 * np.random.random()
    return W
"""
Rewire intra-community edges with a probability of p to an edge between communities, target community and 
excitatory neurons within that community are chosen randomly. 

Inputs:
W - weight matrix of edges in the network
inhibitory_neurons - number of inhibitory neurons (200)
excitatory_neurons - number of excitatory neurons (800)
p - rewiring probability
nr_populations - number of populations (8)
"""
def rewire(W, excitatory_neurons, inhibitory_neurons, p, nr_populations):
    neurons_per_population = int(excitatory_neurons / nr_populations)
    for population in range(nr_populations):
        for it in range(neurons_per_population):
            for it2 in range(neurons_per_population):
                neuron1 = it + population * neurons_per_population
                neuron2 = it2 + population * neurons_per_population
                if W[neuron1][neuron2] != 0 and p >= np.random.random():
                    new_population = np.random.randint(0, nr_populations)
                    while new_population == population:
                        new_population = np.random.randint(0, nr_populations)
                    new_neuron = np.random.randint(0, neurons_per_population) + new_population * neurons_per_population
                    while W[neuron1][new_neuron] != 0:
                        new_neuron = np.random.randint(0, neurons_per_population) + new_population * neurons_per_population
                    W[neuron1][new_neuron] = W[neuron1][neuron2]
                    W[neuron1][neuron2] = 0

"""
Plots connection matrix of the network.

Inputs:
inhibitory_neurons - number of inhibitory neurons (200)
excitatory_neurons - number of excitatory neurons (800)
nr_populations - number of populations (8)
"""
def plot(inhibitory_neurons, excitatory_neurons, nr_populations):
    W = initialize_weights(inhibitory_neurons, excitatory_neurons, nr_populations)
    rewire(W, excitatory_neurons, inhibitory_neurons, p, nr_populations)
    n = inhibitory_neurons + excitatory_neurons
    W_p = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if W[i][j] != 0:
                W_p[i][j] = 1
    plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(W_p, cmap='binary')
    plt.title("Connection Matrix")
    plt.savefig("connection-matrix-" + str(p) + ".svg", format="svg")
    plt.show()
    return W

"""
Initialize the delays between the connections of neurons in the network.
Excitatory to Excitatory - the delay is randomly assigned in the interval [1, 20]
All other neuron connections have a delay of 1.

Inputs:
W - weight matrix of edges in the network
inhibitory_neurons - number of inhibitory neurons (200)
excitatory_neurons - number of excitatory neurons (800)
"""
def get_delays(W, inhibitory_neurons, excitatory_neurons):
    n = inhibitory_neurons + excitatory_neurons
    D = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if W[i][j] != 0:
                if i < excitatory_neurons and j < excitatory_neurons:
                    D[i][j] = int(np.random.randint(1, 21))
                else:
                    D[i][j] = int(1)
            else:
                D[i][j] = 1
    return D

"""
Inject some external current at every time step to ensure activation and prevent it from dying out. Current is injected according to
a Poisson distribution, a number bigger than 0 representing an additional I = 15 current injected.

Inputs:
inhibitory_neurons - number of inhibitory neurons (200)
excitatory_neurons - number of excitatory neurons (800) 
"""

def background_firing(inhibitory_neurons, excitatory_neurons):
    n = inhibitory_neurons + excitatory_neurons
    x = np.random.poisson(lam=0.01, size=n)
    I = np.zeros(n)
    for i in range(n):
        if x[i] > 0:
            I[i] = 15
    return I

"""
Performs the Euler method for simulating neuronal activity. The threshold for spiking is at 30. Additionally the method 
also produces the raster plot of firing neurons.

Inputs:
T - time for each simulation it should run (ms)
Network - the Izhikevich model on which the simulation will run
inhibitory_neurons - number of inhibitory neurons (200)
excitatory_neurons - number of excitatory neurons (800) 
"""

def simulation(T, Network, inhibitory_neurons, excitatory_neurons):
    n = inhibitory_neurons + excitatory_neurons
    V = np.zeros((T, n))
    for t in range(T):
        Network.setCurrent(background_firing(inhibitory_neurons, excitatory_neurons))
        Network.update()
        V[t, :], _ = Network.getState()
    t, m = np.where(V > 29)
    mat = np.zeros((T, n))
    for i in range(len(t)):
        if m[i] < 800:
            mat[t[i], m[i]] = 1
    x, y = np.nonzero(mat)
    fig, axs = plt.subplots(1, 1, figsize=(16, 4))
    axs.set_xticks(list(range(0,1001,100)))
    axs.set_yticks(list(range(0, 801, 200)))
    axs.invert_yaxis()
    axs.set_xlabel("Time (ms)")
    axs.set_ylabel("Neuron number")
    axs.set_title("p = " + str(p))
    axs.scatter(x, y, s=20)
    plt.xlim(left=0)
    plt.savefig("raster-plot-firing-" + str(p) + ".svg", format="svg")
    plt.show()
    return V

"""
Computes the firing rate of each population with sliding window and downsampling.

Inputs:
V - voltage of each neuron at each timestep as generated from the simulation
excitatory_neurons - number of excitatory neurons (800)
nr_populations - number of populations (8)
window_size - the size of the sliding window to compute the mean firing rate
step_size - number by which the sliding window should move each time
"""

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

"""
Plots firing rate per population.

Inputs:
firing_rates - firing rates per population and timestep calculated above
nr_populations - number of populations (8)
step_size - number by which the sliding window should move each time
"""

def plot_firing_rates(firing_rates, nr_populations, step_size=20):
    time_points = np.arange(0, len(firing_rates[0]) * step_size, step_size)
    plt.figure(figsize=(16, 4))

    for pop in range(nr_populations):
        plt.plot(time_points, firing_rates[pop], label=f'Module {pop + 1}')

    plt.xlabel('Time (ms)')
    plt.xticks(list(range(0, 1001, 100)))
    plt.ylabel('Mean Firing Rate (per 50ms window)')
    plt.title('Mean Firing Rate in Each Module')
    plt.legend()
    plt.savefig("mean-firing-rate-" + str(p) + ".svg", format="svg")
    plt.show()

"""
Function by which the plots present in the report were generated for a given p. It initializes the constants for 
the neuron model initializes the network and runs the simulation.

Inputs:
inhibitory_neurons - number of inhibitory neurons (200)
excitatory_neurons - number of excitatory neurons (800)
nr_populations - number of populations (8)
p - rewiring probability
D_max - maximum delay
T - the time that the simulation to run for
"""

def generate_plots(inhibitory_neurons, excitatory_neurons, nr_populations, p, D_max, T):
    n = inhibitory_neurons + excitatory_neurons
    W = plot(inhibitory_neurons, excitatory_neurons, nr_populations)
    D = get_delays(W, inhibitory_neurons, excitatory_neurons)

    a = np.zeros(1000)
    b = np.zeros(1000)
    c = np.zeros(1000)
    d = np.zeros(1000)
    for i in range(1000):
        a[i] = 0.02
        c[i] = -65
        r = np.random.random()
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

    V = simulation(T, Network, inhibitory_neurons, excitatory_neurons)
    firing_rates = compute_firing_rate(V, excitatory_neurons, nr_populations)
    plot_firing_rates(firing_rates, nr_populations)


# list of rewiring probabilities for which simulation and plots to be generated
p_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
for p in p_list:
    generate_plots(200, 800, 8, p, 20, 1000)
import matplotlib.pyplot as plt
import math
from biological_neuron import BiologicalNeuron
from euler_estimator import EulerEstimator
from directed_graph import DirectedGraph


class BiologicalNeuralNetwork:
    def __init__(self, neurons, synapses):
        self.synapses = synapses
        self.neurons = neurons
        self.network = DirectedGraph(synapses, neurons)

    def get_derivatives(self):
        derivatives = []

        for j, n in enumerate(self.neurons):
            ps = [j.index for j in self.network.nodes[j].parents]
            derivatives += [
                (lambda t, x, i=j, n=n, ps=ps: n.derivatives[0](
                    t, x[i*4:(i+1)*4]) + sum(x[p*4] for p in ps)/n.C),
                (lambda t, x, i=j, n=n: n.derivatives[1](t, x[i*4:(i+1)*4])),
                (lambda t, x, i=j, n=n: n.derivatives[2](t, x[i*4:(i+1)*4])),
                (lambda t, x, i=j, n=n: n.derivatives[3](t, x[i*4:(i+1)*4])),
            ]
        return derivatives

    def get_starting_point(self):
        return 0, sum((list(n.starting_point[1]) for n in self.neurons), [])


if __name__ == "__main__":
    def electrode_voltage(t):
        if t > 10 and t < 11:
            return 150
        elif t > 20 and t < 21:
            return 150
        elif t > 30 and t < 40:
            return 150
        elif t > 50 and t < 51:
            return 150
        elif t > 53 and t < 54:
            return 150
        elif t > 56 and t < 57:
            return 150
        elif t > 59 and t < 60:
            return 150
        elif t > 62 and t < 63:
            return 150
        elif t > 65 and t < 66:
            return 150
        return 0
    neuron_0 = BiologicalNeuron(stimulus=electrode_voltage)
    neuron_1 = BiologicalNeuron()
    neuron_2 = BiologicalNeuron()
    neurons = [neuron_0, neuron_1, neuron_2]
    synapses = [(0, 1), (1, 2)]
    network = BiologicalNeuralNetwork(neurons, synapses)
    euler = EulerEstimator(
        derivatives=network.get_derivatives(),
        point=network.get_starting_point()
    )
    plt.plot([n/2 for n in range(160)], [electrode_voltage(n/2)
                                         for n in range(160)])
    euler.plot([0, 80], step_size=0.001,
               filename="biological_neural_network")

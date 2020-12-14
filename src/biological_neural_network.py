import matplotlib.pyplot as plt
import math
from biological_neuron import BiologicalNeuron
from euler_estimator import EulerEstimator
from directed_graph import DirectedGraph


def merge_dicts(dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


class BiologicalNeuralNetwork:
    def __init__(self, neurons, synapses):
        self.synapses = synapses
        self.neurons = neurons
        self.network = DirectedGraph(synapses, neurons)

    def get_derivatives(self):
        derivatives = {}

        for j, n in enumerate(self.neurons):
            ps = [j.index for j in self.network.nodes[j].parents]
            getx = (lambda x, i: {k[:-1]: x[k]
                                  for k in [f"V{i}", f"n{i}", f"m{i}", f"h{i}"]})

            derivatives.update({
                f"V{j}": (lambda t, x, i=j, n=n, ps=ps, getx=getx: n.derivatives['V'](
                    t, getx(x, i)) + sum(x[f"V{p}"] for p in ps)/n.C),
                f"n{j}": (lambda t, x, i=j, n=n, getx=getx: n.derivatives['n'](t, getx(x, i))),
                f"m{j}": (lambda t, x, i=j, n=n, getx=getx: n.derivatives['m'](t, getx(x, i))),
                f"h{j}": (lambda t, x, i=j, n=n, getx=getx: n.derivatives['h'](t, getx(x, i))),
            })
        return derivatives

    def get_starting_point(self):
        return 0, merge_dicts({f"{k}{i}": v for k, v in n.starting_point[1].items()} for i, n in enumerate(self.neurons))


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

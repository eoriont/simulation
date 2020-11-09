import sys
import matplotlib.pyplot as plt
sys.path.append("src")
try:
    from biological_neural_network import BiologicalNeuralNetwork
    from biological_neuron import BiologicalNeuron
    from euler_estimator import EulerEstimator
except ImportError as e:
    print(e)


if __name__ == "__main__":
    def electrode_voltage(t):
        if 10 < t < 11 or \
                20 < t < 21 or \
                30 < t < 40 or \
                50 < t < 51 or \
                53 < t < 54 or \
                56 < t < 57 or \
                59 < t < 60 or \
                62 < t < 63 or \
                65 < t < 66:
            return 50
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
               filename="3_neuron_network")

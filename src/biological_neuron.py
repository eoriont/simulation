import matplotlib.pyplot as plt
import math
import sys
sys.path.append('src')
try:
    from euler_estimator import EulerEstimator
except:
    print("Import failure")


class BiologicalNeuron:
    def __init__(self, stimulus=lambda _: 0):
        self.stimulus = stimulus
        self.derivatives = self.get_derivatives()
        self.starting_point = self.get_starting_point()
        self.estimator = EulerEstimator(self.derivatives, self.starting_point)

    def plot_activity(self, filename):
        plt.plot([n/2 for n in range(160)], [s_t(n/2)
                                             for n in range(160)], zorder=1)
        self.estimator.plot([0, 80], step_size=0.02,
                            filename=filename)

    def get_starting_point(self):
        # h(0)
        x = 0.07 * (math.e**3 + 1)
        h_0 = x/(x+1)

        # n(0)
        n_0 = 1/(1.25*(math.e-1)+1)

        # m(0)
        m_0 = 2.5/(2.5+4*(math.e**2.5 - 1))

        # V(0)
        V_0 = 0
        return 0, (V_0, n_0, m_0, h_0)

    def get_derivatives(self):
        return [self.dV(), BiologicalNeuron.dn, BiologicalNeuron.dm, BiologicalNeuron.dh]

    @staticmethod
    def a_n(t, x):
        return 0.01*(10-x[0])/(math.exp(0.1*(10-x[0]))-1)

    @staticmethod
    def b_n(t, x):
        return 0.125*math.exp(-x[0]/80)

    @staticmethod
    def a_m(t, x):
        return 0.1*(25-x[0])/(math.exp(0.1*(25-x[0]))-1)

    @staticmethod
    def b_m(t, x):
        return 4*math.exp(-x[0]/18)

    @staticmethod
    def a_h(t, x):
        return 0.07 * math.exp(-x[0]/20)

    @staticmethod
    def b_h(t, x):
        return 1/(math.exp(0.1*(30-x[0]))+1)

    @staticmethod
    def dn(t, x):
        return BiologicalNeuron.a_n(t, x)*(1 - x[1]) - BiologicalNeuron.b_n(t, x)*x[1]

    @staticmethod
    def dm(t, x):
        return BiologicalNeuron.a_m(t, x)*(1 - x[2]) - BiologicalNeuron.b_m(t, x)*x[2]

    @staticmethod
    def dh(t, x):
        return BiologicalNeuron.a_h(t, x)*(1 - x[3]) - BiologicalNeuron.b_h(t, x)*x[3]

    C = 1
    Vna = 115
    Vk = -12
    VL = 10.6
    _gna = 120
    _gk = 36
    _gl = 0.3

    @staticmethod
    def INa(t, x):
        return BiologicalNeuron.gna(t, x)*(x[0]-BiologicalNeuron.Vna)

    @staticmethod
    def gna(t, x):
        return BiologicalNeuron._gna * x[2]**3 * x[3]

    @staticmethod
    def Ik(t, x):
        return BiologicalNeuron.gk(t, x)*(x[0]-BiologicalNeuron.Vk)

    @staticmethod
    def gk(t, x):
        return BiologicalNeuron._gk * x[1]**4

    @staticmethod
    def IL(t, x):
        return BiologicalNeuron.gl(t, x)*(x[0]-BiologicalNeuron.VL)

    @staticmethod
    def gl(t, x):
        return BiologicalNeuron._gl

    def dV(self):
        return lambda t, x: 1/BiologicalNeuron.C * (self.stimulus(t) - BiologicalNeuron.INa(t, x) - BiologicalNeuron.Ik(t, x) - BiologicalNeuron.IL(t, x))


if __name__ == "__main__":
    def s_t(t):
        if 10 <= t <= 11 or 20 <= t <= 21 or \
                30 <= t <= 40 or 50 <= t <= 51 or \
                53 <= t <= 54 or 56 <= t <= 57 or \
                59 <= t <= 60 or 62 <= t <= 63 or \
                65 <= t <= 66:
            return 150
        return 0
    neuron = BiologicalNeuron(s_t)
    neuron.plot_activity("neuron_activity.png")

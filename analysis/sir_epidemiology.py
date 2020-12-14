import sys
sys.path.append('src')
try:
    from euler_estimator import EulerEstimator
except:
    print("Import failure")

derivatives = {
    'susceptible': (lambda t, x: -0.0003*x['susceptible']*x['infected']),
    'infected': (lambda t, x: 0.0003*x['susceptible']*x['infected'] - 0.02*x['infected']),
    'recovered': (lambda t, x: 0.02*x['infected'])
}
starting_point = (0, {'susceptible': 1000, 'infected': 1, 'recovered': 0})

estimator = EulerEstimator(derivatives, starting_point)

estimator.plot([0, 365], step_size=0.001,
               filename="sir_epidemiology.png")

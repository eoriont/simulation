import sys
sys.path.append('src')
try:
    from euler_estimator import EulerEstimator
except:
    print("Import failure")

derivatives = {
    'prey': (lambda t, x: 0.6*x['prey'] - 0.05*x['prey']*x['predator']),
    'predator': (lambda t, x: -0.9*x['predator'] + 0.02*x['prey']*x['predator'])
}
starting_point = (0, {'prey': 100, 'predator': 10})

estimator = EulerEstimator(derivatives, starting_point)

estimator.plot([0, 100], step_size=0.001,
               filename="predator_prey_plot.png")

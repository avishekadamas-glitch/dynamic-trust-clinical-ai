#!/usr/bin/env python3
"""Data-free numerical examples and implementation checks; not model validation."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'work/corrected'))
sys.path.insert(0, str(ROOT / 'work/peer_review/analysis'))
from ordinal_dependence import PairKernel, kernel_check
from full_history import forecast
from check_full_history import static_direct
from predict_corrected import build_features, prefix_check


def main():
    # Two hypothetical distributions over a system's accuracy. Both place half
    # their mass above the 0.8 standard, but have different expected accuracies.
    weights = np.array([0.5, 0.5])
    supports = [np.array([0.7, 0.9]), np.array([0.1, 0.9])]
    capability = [float(weights[s >= 0.8].sum()) for s in supports]
    predictive = [float(weights @ s) for s in supports]
    assert np.allclose(capability, [0.5, 0.5])
    assert np.allclose(predictive, [0.8, 0.5])
    # The predictive interpretation here assumes exchangeable Bernoulli cases.
    actions = ['rely' if p > 0.65 else 'override' for p in predictive]
    assert actions == ['rely', 'override']
    a = np.log(3)
    sig = lambda z: 1 / (1 + np.exp(-z))
    ordered = [float(sig(-a / 2)), float(sig(a / 2))]
    assert np.isclose(ordered[0], 1 / (1 + np.sqrt(3)))
    assert np.isclose(sum(ordered), 1)

    rng = np.random.default_rng(20261002)
    synthetic = rng.integers(0, 7, size=(2, 8, 21))
    kernel = PairKernel(synthetic)
    _, probabilities = kernel.evaluate(0.3, 0.4, gradients=False)
    assert np.min(probabilities) >= 0
    normalization_error = float(abs(probabilities.sum(axis=(1, 2)) - 1).max())
    assert normalization_error < 1e-12
    loc = np.array([0.3, 0.4]); step = 1e-5
    _, analytic = kernel.evaluate(*loc)
    finite = np.array([(kernel.evaluate(*(loc + np.eye(2)[j] * step))[0]
                       - kernel.evaluate(*(loc - np.eye(2)[j] * step))[0]) / (2 * step)
                      for j in range(2)])
    gradient_error = float(abs(analytic - finite).max())
    assert gradient_error < 1e-5

    y = rng.integers(0, 7, 21)
    margins = np.tile(np.full(7, 1 / 7), (21, 1))
    dynamic, _ = forecast(y, margins, eta=0.3, phi=0.4, nb=61, nu=32)
    assert np.all(dynamic >= 0) and np.allclose(dynamic.sum(axis=1), 1, atol=1e-10)
    changed = y.copy(); changed[8:] = (changed[8:] + 1) % 7
    alternative, _ = forecast(changed, margins, eta=0.3, phi=0.4, nb=61, nu=32)
    prefix_error = float(abs(dynamic[:8] - alternative[:8]).max())
    assert prefix_error < 1e-12
    static, _ = forecast(y, margins, eta=0.3, phi=0.0, nb=121, nu=64)
    direct = static_direct(y, margins, eta=0.3, nodes=241)
    static_error = float(abs(static - direct).max())
    assert static_error < 1e-7

    ratings = synthetic.reshape(16, 21) + 1
    conditions = np.repeat([1, 2], 8)
    ids = np.arange(16)
    matrix, _, people, occasions = build_features(ratings, conditions, ids)
    checked = prefix_check(ratings, conditions, ids, matrix, people, occasions)
    result = {'status': 'pass', 'data': 'synthetic only; no clinical input read',
              'capability_probabilities': capability, 'example_predictions': predictive,
              'example_actions': actions, 'evidence_order_final_probabilities': ordered,
              'pair_probability_normalization_error': normalization_error,
              'analytic_gradient_error': gradient_error,
              'full_history_prefix_error': prefix_error,
              'static_integration_error': static_error,
              'multinomial_prefix_checks': checked,
              'bivariate_kernel_max_error': kernel_check()['maximum_absolute_difference_vs_scipy']}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

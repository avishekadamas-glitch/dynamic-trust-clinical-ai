#!/usr/bin/env python3
"""Reproduce the fresh independent conditional-probability integration checks.

Run from any directory; the archive root is inferred from this file's location.
This is the exact adaptive-bivariate and direct-static calculation used for the
2026-09-07 check. No models are fitted and no production filter is imported.
Dependencies: numpy, pandas, scipy; Python standard library.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import ndtr, ndtri, roots_hermitenorm

BASE = Path(__file__).resolve().parents[2]
P = BASE
r = pd.read_csv(P / 'upload/Corrected_Clinical_Trial_Long_Data.csv')
fits = pd.read_csv(P / 'work/corrected/dependence/copula_lopo_fits.csv').set_index('ID')
pred = pd.read_csv(P / 'work/peer_review/analysis/full_history_481_64.csv')
cols = [f'p{i}' for i in range(1, 8)]
out = {}
errs = []
serrs = []
merrs = []
for ident, f in fits.iterrows():
    person = r[r.ID == ident].sort_values('Trial')
    g = person.CaseSeries.iloc[0]
    train = r[(r.CaseSeries == g) & (r.ID != ident)]
    tab = pd.crosstab(train.Trial, train.direct_trust_item).reindex(
        index=range(1, 22), columns=range(1, 8), fill_value=0).to_numpy()
    probs = (tab + .5) / 36.5
    cuts = np.c_[-np.inf * np.ones(21), ndtri(probs.cumsum(1)[:, :6]), np.inf * np.ones(21)]
    y = person.direct_trust_item.to_numpy(int) - 1
    for mode, eta, phi in [('static', f.omega_static, 0.),
                           ('dynamic', f.omega_dynamic, f.phi_dynamic)]:
        rho = eta + (1 - eta) * phi
        sd = np.sqrt(1 - rho * rho)
        lo, hi = cuts[0, y[0]:y[0] + 2]
        q = []
        for k in range(7):
            v = quad(lambda z: np.exp(-z * z / 2) / np.sqrt(2 * np.pi) *
                     (ndtr((cuts[1, k + 1] - rho * z) / sd) -
                      ndtr((cuts[1, k] - rho * z) / sd)),
                     lo, hi, epsabs=1e-13, epsrel=1e-13)[0] / probs[0, y[0]]
            q.append(v)
        saved = pred[(pred.ID == ident) & (pred.model == mode + '_full_history')].sort_values('Trial')
        errs.append(float(abs(np.asarray(q) - saved.iloc[0][cols].to_numpy(float)).max()))
    # Direct one-dimensional static posterior update, independently coded at 801 nodes.
    x, w = roots_hermitenorm(801)
    b = x * np.sqrt(f.omega_static)
    w = w / np.sqrt(2 * np.pi)
    static = []
    sd = np.sqrt(1 - f.omega_static)
    for t in range(21):
        ps = ndtr((cuts[t, 1:, None] - b) / sd) - ndtr((cuts[t, :-1, None] - b) / sd)
        if t:
            static.append(ps @ w)
        w *= ps[y[t]]
        w /= w.sum()
    saved = pred[(pred.ID == ident) & (pred.model == 'static_full_history')].sort_values('Trial')[cols].to_numpy()
    serrs.append(float(abs(np.array(static) - saved).max()))
    # Independently verify that the saved baseline margins exclude the held-out person.
    margin_saved = pd.read_csv(P / 'work/corrected/dependence/copula_lopo_predictions.csv')
    margin_saved = margin_saved[(margin_saved.ID == ident) &
                                (margin_saved.model == 'case_margin')].sort_values('Trial')[cols].to_numpy()
    merrs.append(float(abs(probs[1:] - margin_saved).max()))
out = {
    'independent_adaptive_bivariate_first_forecasts': 136,
    'maximum_first_forecast_probability_error': max(errs),
    'independent_direct_static_forecasts': 1360,
    'maximum_static_probability_error_801_nodes': max(serrs),
    'independent_training_only_case_margins': 1360,
    'maximum_case_margin_error': max(merrs),
}
(P / 'work/final_verification/statistics_independent_integration.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))

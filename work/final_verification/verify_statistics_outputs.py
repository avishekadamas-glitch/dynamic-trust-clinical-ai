#!/usr/bin/env python3
"""Recompute the independent final statistical-output verification.

Run from any directory; the archive root is inferred from this file's location.
This reproduces the calculations executed on 2026-09-07 without fitting models.
Dependencies: numpy, pandas; Python standard library.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
P = BASE
A = P / 'work/peer_review/analysis'
C = P / 'work/corrected'
r = pd.read_csv(P / 'upload/Corrected_Clinical_Trial_Long_Data.csv')
key = pd.read_csv(P / 'upload/Appendix_A_Actual_Sequence.csv')
ans = {}
ans['raw_counts'] = {
    'rows': len(r), 'people': r.ID.nunique(),
    'series_people': r.groupby('CaseSeries').ID.nunique().to_dict(),
    'unique_keys': not r.duplicated(['ID', 'Trial']).any(),
    'nurse_physician': r.drop_duplicates('ID').Job_Category.value_counts().to_dict(),
    'trial_sets_correct': r.groupby('ID').Trial.apply(lambda v: sorted(v) == list(range(1, 22))).all().item(),
    'missing_ratings': int(r.direct_trust_item.isna().sum()),
    'missing_decisions': int(r.decision_accept.isna().sum()),
    'missing_decision_people': r[r.decision_accept.isna()].ID.nunique(),
    'missing_decision_series': r[r.decision_accept.isna()].CaseSeries.unique().tolist(),
}
k = r.merge(key[['CaseSeries', 'Trial', 'RawTaskNumber', 'Accuracy']],
            on=['CaseSeries', 'Trial'], suffixes=['', '_key'], validate='many_to_one')
ans['key_match'] = {name: bool((k[name] == k[name + '_key']).all())
                    for name in ['RawTaskNumber', 'Accuracy']}
ans['key_correctness_counts'] = {str(g): d.Accuracy.value_counts().to_dict()
                                 for g, d in key.groupby('CaseSeries')}
keys = key.pivot(index='Trial', columns='CaseSeries', values='Accuracy')
ans['series_correctness_differences'] = keys.index[keys.A != keys.B].tolist()
ans['decision_counts'] = [
    {'series': g, 'Accuracy': a, 'accept': int((d.decision_accept == 1).sum()),
     'reject': int((d.decision_accept == 0).sum()), 'missing': int(d.decision_accept.isna().sum())}
    for (g, a), d in r.groupby(['CaseSeries', 'Accuracy'])
]
y = r.pivot(index=['CaseSeries', 'ID'], columns='Trial', values='direct_trust_item').sort_index().to_numpy(int)
counts = np.stack([(y == cat).sum(axis=1) for cat in range(1, 8)], 1)
obs = {
    'adjacent_equal': float((y[:, 1:] == y[:, :-1]).mean()),
    'distinct_categories': float((counts > 0).sum(axis=1).mean()),
    'modal_share': float((counts.max(axis=1) / 21).mean()),
    'constant_trajectories': float((counts.max(axis=1) == 21).sum()),
}
for g, slc in [('A', slice(0, 34)), ('B', slice(34, 68))]:
    obs['adjacent_equal_' + g] = float((y[slc, 1:] == y[slc, :-1]).mean())
for lag in [2, 5, 10]:
    obs['equal_lag_' + str(lag)] = float((y[:, lag:] == y[:, :-lag]).mean())
obs['adjacent_equal_early'] = float((y[:, 1:11] == y[:, :10]).mean())
obs['adjacent_equal_late'] = float((y[:, 11:] == y[:, 10:-1]).mean())
ans['observed_summaries'] = obs
ans['identical_adjacent_count'] = int((y[:, 1:] == y[:, :-1]).sum())
checks = pd.read_csv(A / 'model_checks.csv').set_index('summary')
sims = pd.read_csv(A / 'model_check_replicates.csv')
ans['model_check_verification'] = {
    'replicates': len(sims),
    'max_observed_error': max(abs(checks.loc[k, 'observed'] - v) for k, v in obs.items()),
    'max_mean_error': float(abs(sims.mean() - checks.model_mean).max()),
    'max_lower_error': float(abs(sims.quantile(.025) - checks.model_range_lower).max()),
    'max_upper_error': float(abs(sims.quantile(.975) - checks.model_range_upper).max()),
    'constant_trajectories_sum': int(sims.constant_trajectories.sum()),
    'constant_trajectories_exact_mean': int(sims.constant_trajectories.sum()) / len(sims),
}
boot = pd.read_csv(C / 'dependence/bootstrap_replicates.csv')
rec = pd.read_csv(C / 'dependence/recovery_replicates.csv')
ans['parameter_bootstrap'] = {
    'replicates': len(boot), 'eta': boot.omega.quantile([.025, .975]).tolist(),
    'phi': boot.phi.quantile([.025, .975]).tolist(), 'converged': int(boot.optimizer_success.sum()),
    'boundary_counts': boot.filter(regex='boundary$').sum().to_dict(),
}
ans['recovery'] = [
    {'true_phi': ph, 'n': len(g), 'phi_mean': g.phi.mean(),
     'phi_rmse': float(np.sqrt(np.mean((g.phi - ph) ** 2))), 'eta_mean': g.omega.mean(),
     'eta_rmse': float(np.sqrt(np.mean((g.omega - .3) ** 2))),
     'phi_boundary': g.phi_lower_boundary.mean(), 'eta_boundary': g.omega_lower_boundary.mean(),
     'converged': int(g.optimizer_success.sum())}
    for ph, g in rec.groupby('true_phi')
]
a = pd.read_csv(C / 'dependence/copula_lopo_predictions.csv')
b = pd.read_csv(A / 'full_history_481_64.csv')
m = pd.read_csv(C / 'prediction/corrected_predictions.csv').rename(columns={
    'observed_rating': 'observed_category', 'normalized_RPS': 'mean_normalized_RPS',
    **{f'p_rating_{i}': f'p{i}' for i in range(1, 8)},
})
m['model'] = 'multinomial_C' + m.C.astype(str) + '_' + m.model
p = pd.concat([a, b, m])
cols = [f'p{i}' for i in range(1, 8)]
v = p[cols].to_numpy()
targets = p.observed_category.to_numpy(int)
p['LL_new'] = -np.log(v[np.arange(len(p)), targets - 1])
p['RPS_new'] = np.square(np.cumsum(v, axis=1)[:, :6] -
                          (targets[:, None] <= np.arange(1, 7))).mean(axis=1)
actual = r.set_index(['ID', 'Trial']).direct_trust_item
ans['score_input_checks'] = {
    'records': len(p), 'models': p.model.nunique(),
    'counts_each_model': p.groupby('model').size().to_dict(),
    'counts_each_person_model': sorted(p.groupby(['ID', 'model']).size().unique().tolist()),
    'duplicate_keys': int(p.duplicated(['ID', 'Trial', 'model']).sum()),
    'target_error': float(np.max(np.abs(p.set_index(['ID', 'Trial']).observed_category -
                                       actual.reindex(p.set_index(['ID', 'Trial']).index)))),
    'min_probability': float(v.min()),
    'max_probability_sum_error': float(abs(v.sum(1) - 1).max()),
    'max_saved_logloss_error': float(abs(p.LL_new - p.log_loss).max()),
    'max_saved_rps_error': float(abs(p.RPS_new - p.mean_normalized_RPS).max()),
}
means = p.groupby('model')[['LL_new', 'RPS_new']].mean().rename(columns={
    'LL_new': 'log_loss', 'RPS_new': 'mean_normalized_RPS',
})
saved = pd.read_csv(A / 'all_forecast_scores.csv').set_index('model')
ans['score_means'] = means.reset_index().to_dict('records')
ans['max_aggregate_score_error'] = float(abs(means - saved).max().max())
pp = p.groupby(['CaseSeries', 'ID', 'model'])[['LL_new', 'RPS_new']].mean()
savedc = pd.read_csv(A / 'all_paired_score_contrasts.csv')
rng = np.random.default_rng(202609063)
ia = rng.integers(0, 34, size=(10000, 34))
ib = rng.integers(0, 34, size=(10000, 34))
contrast_errors = []
contrasts = []
for row in savedc.itertuples():
    metric = 'LL_new' if row.metric == 'log_loss' else 'RPS_new'
    w = pp[metric].unstack('model')
    delta = w[row.comparator] - w[row.reference]
    da = delta.loc['A'].to_numpy()
    db = delta.loc['B'].to_numpy()
    samples = (da[ia].sum(1) + db[ib].sum(1)) / 68
    ci = np.percentile(samples, [2.5, 97.5])
    sci = json.loads(row.interval95)
    contrast_errors.extend([abs(delta.mean() - row.mean_difference), *abs(ci - sci),
                            abs(da.mean() - row.series_A_difference),
                            abs(db.mean() - row.series_B_difference)])
    contrasts.append({'metric': row.metric, 'comparator': row.comparator,
                      'reference': row.reference, 'difference': delta.mean(), 'CI': ci.tolist()})
ans['paired_contrast_verification'] = {
    'comparisons': len(savedc), 'max_saved_error': float(max(contrast_errors)),
}
ans['contrasts'] = contrasts
# Rank uses independently built 40 cell indicators on target trials.
z = r[r.Trial >= 2].copy()
design = pd.get_dummies(z.CaseSeries + '_' + z.Trial.astype(str), dtype=float).to_numpy()
cur = (z.Accuracy == 'Correct').to_numpy(float)
prev = (r.sort_values(['ID', 'Trial']).groupby('ID').Accuracy.shift()
        .loc[z.index].eq('Correct').to_numpy(float))
ans['design_ranks'] = [int(np.linalg.matrix_rank(m))
                        for m in [design, np.c_[design, cur], np.c_[design, prev]]]
# Compare all complete grids using explicitly keyed alignment.
bidx = b.set_index(['ID', 'Trial', 'model'])
grids = {}
for name, path in [('241_64', A / 'full_history_241_64.csv'),
                   ('241_96', P / 'work/peer_review/verification/full_history_241_96.csv')]:
    if path.exists():
        z = pd.read_csv(path).set_index(['ID', 'Trial', 'model']).reindex(bidx.index)
        grids[name] = {
            'max_probability_difference': float(abs(bidx[cols] - z[cols]).max().max()),
            'max_trial_logloss_difference': float(abs(bidx.log_loss - z.log_loss).max()),
        }
ans['grids'] = grids
# Independently verify first-forecast full/one-lag agreement after explicit keys.
first = []
for label in ['static', 'dynamic']:
    lhs = b[(b.Trial == 2) & (b.model == label + '_full_history')].set_index('ID')[cols]
    rhs = a[(a.Trial == 2) & (a.model == label + '_one_lag')].set_index('ID')[cols]
    first.append({'model': label, 'max_error': float(abs(lhs - rhs).max().max())})
ans['first_lag'] = first
fit = pd.read_csv(C / 'dependence/copula_lopo_fits.csv')
ans['saved_lopo_fits'] = {
    'rows': len(fit), 'ID_unique': fit.ID.is_unique,
    'min_dynamic_eta': fit.omega_dynamic.min(), 'max_dynamic_eta': fit.omega_dynamic.max(),
    'min_phi': fit.phi_dynamic.min(), 'max_phi': fit.phi_dynamic.max(),
    'all_static_converged': bool(fit.static_converged.all()),
    'all_dynamic_converged': bool(fit.dynamic_converged.all()),
}
mfits = json.loads((C / 'prediction/corrected_prediction_summary.json').read_text())['fit_diagnostics']
ans['multinomial_fit_diagnostics'] = {
    'fits': len(mfits), 'warning_count': sum(len(x['warnings']) for x in mfits),
    'max_iterations': max(x['iterations'] for x in mfits),
}
(P / 'work/final_verification/statistics_checks.json').write_text(json.dumps(ans, indent=2))
for k, v in ans.items():
    if k not in ['contrasts', 'score_means']:
        print(k, json.dumps(v))

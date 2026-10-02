#!/usr/bin/env python3
"""Corrected direct-item sequential prediction; no latent-trust identification.

Run: python predict_corrected.py --input Corrected_Clinical_Trial_Long_Data.csv
     --sequence Appendix_A_Actual_Sequence.csv --output-dir results

Only ID, CaseSeries, Trial, and direct_trust_item enter prediction. Sequence fields
are checked against the supplied design key; all centered and composite columns
are deliberately excluded. Python/pandas are used for analysis, not XLSX writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


SEED = 20260905
MODEL_NAMES = ("case_series", "plus_past_distribution", "plus_latest_rating")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_features(ratings, condition, participant_ids):
    """Each row uses a predeclared design cell and strictly preceding ratings."""
    n_people, n_times = ratings.shape
    rows, ys, people, occasions = [], [], [], []
    for person in range(n_people):
        previous_counts = np.zeros(7)
        for t in range(1, n_times):
            # Before occasion t+1, add only the newly completed occasion t.
            previous_counts[ratings[person, t - 1] - 1] += 1
            context = np.zeros(40)
            context[(int(condition[person]) - 1) * 20 + (t - 1)] = 1
            history = previous_counts / t
            latest = np.zeros(7)
            latest[ratings[person, t - 1] - 1] = 1
            rows.append(np.r_[context, history, latest])
            ys.append(ratings[person, t])
            people.append(participant_ids[person])
            occasions.append(t + 1)
    matrix = np.asarray(rows)
    return matrix, np.asarray(ys), np.asarray(people), np.asarray(occasions)


def prefix_check(ratings, condition, participant_ids, matrix, people, occasions):
    """For all 20 targets and 68 people, perturb current and all future labels."""
    checked = 0
    for target_occasion in range(2, 22):
        changed = ratings.copy()
        # Cyclic relabeling changes every selected outcome, including category 4.
        changed[:, target_occasion - 1:] = changed[:, target_occasion - 1:] % 7 + 1
        altered, *_ = build_features(changed, condition, participant_ids)
        idx = occasions == target_occasion
        if not np.array_equal(matrix[idx], altered[idx]):
            raise AssertionError("Current/future ratings affected current prediction features")
        checked += int(idx.sum())
    return checked


def metrics(y, p):
    # Stable clipping only in logarithmic scoring; predicted probabilities remain original.
    loss = -np.log(np.maximum(p[np.arange(len(y)), y - 1], np.finfo(float).tiny))
    predicted_cdf = np.cumsum(p, axis=1)[:, :6]
    observed_cdf = y[:, None] <= np.arange(1, 7)[None, :]
    rps = np.mean((predicted_cdf - observed_cdf) ** 2, axis=1)
    accuracy = (p.argmax(axis=1) + 1 == y).astype(float)
    return loss, rps, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("upload/Corrected_Clinical_Trial_Long_Data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/corrected/prediction"))
    parser.add_argument("--sequence", type=Path, default=Path("upload/Appendix_A_Actual_Sequence.csv"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    raw = pd.read_csv(args.input)
    # Validate unique and complete participant-by-true-trial keys, not raw task numbers.
    needed = {"ID", "CaseSeries", "Trial", "RawTaskNumber", "Accuracy", "direct_trust_item"}
    if not needed.issubset(raw.columns):
        raise ValueError("Corrected input lacks required fields")
    if raw.duplicated(["ID", "Trial"]).any():
        raise ValueError("Duplicate participant-by-trial records")
    participant_ids = np.sort(raw["ID"].unique())
    n_people = len(participant_ids)
    if n_people != 68 or len(raw) != 1428:
        raise ValueError("Expected 68 participants and 1428 corrected observations")
    if not raw.groupby("ID")["CaseSeries"].nunique().eq(1).all():
        raise ValueError("Participant changes series")
    if not raw.groupby("ID")["Trial"].apply(lambda x: set(x) == set(range(1, 22))).all():
        raise ValueError("Trials must be exactly 1 through 21 for every participant")
    sequence = pd.read_csv(args.sequence)
    key = ["CaseSeries", "Trial"]
    if len(sequence) != 42 or sequence.duplicated(key).any():
        raise ValueError("Expected 42 unique design cells")
    checked = raw.merge(sequence[key + ["RawTaskNumber", "Accuracy"]], on=key,
                        suffixes=("", "_key"), how="left", validate="many_to_one", indicator=True)
    if not checked["_merge"].eq("both").all():
        raise ValueError("Unmatched design cells")
    for field in ["RawTaskNumber", "Accuracy"]:
        if not checked[field].eq(checked[field + "_key"]).all():
            raise ValueError("Corrected data disagree with design key: " + field)
    extracted = raw.pivot(index="ID", columns="Trial", values="direct_trust_item").loc[
        participant_ids, range(1, 22)].to_numpy(dtype=float)
    if not (np.isfinite(extracted).all() and
            ((extracted == np.floor(extracted)) & (extracted >= 1) & (extracted <= 7)).all()):
        raise ValueError("Expected complete integer direct-trust ratings from 1 through 7")
    ratings = extracted.astype(int)
    series = raw.drop_duplicates("ID").set_index("ID").loc[participant_ids, "CaseSeries"]
    if set(series) != {"A", "B"}:
        raise ValueError("Unexpected case series")
    condition = series.map({"A": 1, "B": 2}).to_numpy()
    X, y, participant, occasion = build_features(ratings, condition, participant_ids)
    prefix_checks = prefix_check(ratings, condition, participant_ids, X, participant, occasion)
    labels = np.arange(1, 8)
    rows_predictions, rows_participants, fit_diagnostics = [], [], []
    summaries = []
    results = {}
    configurations = [(1.0, True), (0.1, False), (10.0, False)]
    for C, primary in configurations:
        for model, width in zip(MODEL_NAMES, (40, 47, 54)):
            probabilities = np.full((len(y), 7), np.nan)
            max_iterations = 0
            convergence_warning_count = 0
            for held_out in participant_ids:
                test = participant == held_out
                train = ~test
                assert not set(participant[train]) & set(participant[test])
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    estimator = LogisticRegression(C=C, solver="lbfgs", max_iter=1000,
                                                   tol=1e-4, random_state=SEED)
                    estimator.fit(X[train, :width], y[train])
                conv = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
                convergence_warning_count += len(conv)
                if not np.array_equal(estimator.classes_, labels):
                    raise AssertionError("A training fold lacks one or more response categories")
                probabilities[test] = estimator.predict_proba(X[test, :width])
                iterations = int(np.max(estimator.n_iter_))
                max_iterations = max(max_iterations, iterations)
                fit_diagnostics.append({"C": C, "model": model, "held_out_ID": int(held_out),
                                        "iterations": iterations, "warnings": conv})
            if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
                raise AssertionError("Invalid out-of-fold probability forecasts")
            nll, rps, accuracy = metrics(y, probabilities)
            results[(C, model)] = (nll, rps, accuracy)
            summaries.append({"C": C, "primary": primary, "model": model,
                              "mean_log_loss": float(nll.mean()), "mean_normalized_RPS": float(rps.mean()),
                              "accuracy": float(accuracy.mean()), "max_iterations": max_iterations,
                              "convergence_warnings": convergence_warning_count})
            for person in participant_ids:
                select = participant == person
                rows_participants.append({"C": C, "primary": primary, "model": model,
                                          "ID": int(person), "n_forecasts": int(select.sum()),
                                          "mean_log_loss": float(nll[select].mean()),
                                          "mean_normalized_RPS": float(rps[select].mean()),
                                          "accuracy": float(accuracy[select].mean())})
            for index in range(len(y)):
                row = {"C": C, "primary": primary, "model": model,
                       "ID": int(participant[index]), "Trial": int(occasion[index]),
                       "CaseSeries": str(series.loc[participant[index]]),
                       "observed_rating": int(y[index]), "log_loss": float(nll[index]),
                       "normalized_RPS": float(rps[index])}
                row.update({f"p_rating_{k}": float(probabilities[index, k - 1]) for k in labels})
                rows_predictions.append(row)
            print(f"C={C:g}, {model}: logloss={nll.mean():.6f}, RPS={rps.mean():.6f}, "
                  f"warnings={convergence_warning_count}", flush=True)
    rng = np.random.default_rng(SEED)
    bootstrap_indices = rng.integers(0, n_people, (args.bootstrap, n_people))
    contrasts = []
    for C, primary in configurations:
        for first, second in ((MODEL_NAMES[0], MODEL_NAMES[1]), (MODEL_NAMES[1], MODEL_NAMES[2]),
                              (MODEL_NAMES[0], MODEL_NAMES[2])):
            for metric_name, metric_index in (("log_loss", 0), ("normalized_RPS", 1)):
                difference = results[(C, first)][metric_index] - results[(C, second)][metric_index]
                participant_difference = difference.reshape(n_people, 20).mean(axis=1)
                sampled = participant_difference[bootstrap_indices].mean(axis=1)
                lo, hi = np.quantile(sampled, [.025, .975])
                contrasts.append({"C": C, "primary": primary, "comparison": first + " minus " + second,
                                  "metric": metric_name, "positive_favors": second,
                                  "mean_improvement": float(participant_difference.mean()),
                                  "bootstrap_95_percentile_interval": [float(lo), float(hi)],
                                  "participants_improved": int((participant_difference > 0).sum()),
                                  "participants_tied": int((participant_difference == 0).sum())})
    assumptions = [
        "Trial order and RawTaskNumber/Accuracy mappings agree exactly with the supplied actual-sequence key.",
        "The primary endpoint is direct_trust_item, an observed ordered seven-category response; it is not a probability anchor.",
        "Case-by-series predictors jointly absorb design position, shared content, and condition; these effects are not separately causal.",
        "Neither current nor prior correctness, choice, transparency, likelihood-to-act, difficulty, composites, or supplied centered columns enter the forecasts.",
        "The held-out participant's completed direct ratings are revealed sequentially only as prediction features, never for coefficient fitting.",
        "Forecasts target the next observed rating under the same two fixed sequences; they do not establish transport to unseen cases or sequences.",
        "No capability probability, elapsed-time forgetting rate, learning gain, or construct validity is estimated.",
        "Bootstrap intervals condition on fitted out-of-fold predictions and omit uncertainty from refitting the overlapping training sets.",
        "Ratings were recorded before correctness feedback. Choice and rating appeared on the same screen; their actual order was not verified by timestamps. No current rating is used to predict the same action.",
        "This corrected analysis supersedes predictions from Clean_R3.csv, whose endpoint and trial mapping differed.",
    ]
    summary = {
        "source": {"filename": args.input.name, "sha256": sha256(args.input),
                   "sequence_filename": args.sequence.name, "sequence_sha256": sha256(args.sequence),
                   "endpoint": "direct_trust_item", "rows": int(raw.shape[0]), "columns": int(raw.shape[1]),
                   "participants": n_people, "occasions_per_participant": 21,
                   "forecast_occasions_per_participant": 20, "forecasts_per_model": len(y),
                   "series_counts": {str(k): int(v) for k, v in series.value_counts().items()},
                   "rating_counts": {str(k): int((ratings == k).sum()) for k in labels}},
        "methods": {"model": "ridge multinomial logistic categorical probability forecast",
                    "solver": "lbfgs", "primary_C": 1.0, "sensitivity_C": [0.1, 10.0],
                    "max_iter": 1000, "tolerance": 1e-4,
                    "features": {MODEL_NAMES[0]: 40, MODEL_NAMES[1]: 47, MODEL_NAMES[2]: 54},
                    "validation": "leave-one-participant-out; 68 folds; one-step predictions at occasions 2–21",
                    "primary_metric": "natural logarithmic loss", "secondary_metric": "RPS divided by six",
                    "bootstrap_replicates": args.bootstrap, "seed": SEED,
                    "bootstrap_unit": "participant; paired conditional out-of-fold loss differences",
                    "prefix_leakage_check_passed": True, "participant_target_prefix_checks": prefix_checks,
                    "sequence_join_verified": True, "original_IDs_preserved": True,
                    "note": "C=1 fixed before fitting; other C values are reported regardless of direction; no tuning."},
        "results": summaries, "paired_contrasts": contrasts, "assumptions_and_limits": assumptions,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "script_sha256": sha256(__file__), "elapsed_seconds": time.time() - started,
        "fit_diagnostics": fit_diagnostics,
    }
    (args.output_dir / "corrected_prediction_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(rows_predictions).to_csv(args.output_dir / "corrected_predictions.csv", index=False)
    pd.DataFrame(rows_participants).to_csv(args.output_dir / "corrected_participant_losses.csv", index=False)
    lines = ["# Corrected direct-item sequential prediction", "",
             "This exploratory analysis supersedes the earlier Clean_R3.csv forecasts. It uses the direct trust item and supplied actual trial sequence. It assesses bounded predictive feasibility, not validation of the full mathematical trust framework.", "",
             "## Methods", "",
             "The corrected long data contained 68 clinicians (34 in each case series), each with 21 complete direct-trust ratings coded 1–7. Participant-by-trial records were unique, trials were exactly 1–21 for each participant, and all 1,428 rows matched the supplied 42-cell actual-sequence key on series, trial, raw task number, and supplied correctness label. The analysis retained the supplied participant IDs. It did not use the older Clean_R3.csv ratings or any globally centered, composite, intention, difficulty, or concurrent action fields.", "",
             "We made one-step forecasts at trials 2–21, giving 1,360 forecasts per model. Ridge multinomial logistic regression estimated probabilities for the seven response categories. The baseline contained 40 trial-by-series indicators. The history model additionally contained seven proportions describing the participant's direct ratings on all completed preceding trials. The recency model additionally contained seven indicators for the most recent direct rating. Categorical modeling did not require equal spacing between rating categories or proportional odds. All models used C=1 as the primary regularization setting; C=0.1 and C=10 were prespecified for this computational rerun as sensitivity settings, with no outcome-driven selection. The secondary analysis was exploratory and not preregistered.", "",
             "Leave-one-participant-out validation fitted coefficients on the other 67 participants. Each held-out participant's completed direct ratings became available sequentially as prediction features. No current or future held-out rating entered a forecast. A perturbation check cyclically changed current and future labels for every participant at each target trial; all 1,360 participant-target feature checks passed. The design-cell indicators address forecasts for another participant facing the same two sequences, not generalization to new cases or a new ordering.", "",
             "Natural-log predictive loss was the primary score. The normalized ranked probability score (RPS) averaged squared errors across the six cumulative-category probabilities; lower values indicate better probability forecasts. Accuracy was descriptive. Paired improvements were summarized with 10,000 participant-bootstrap percentile intervals. These intervals condition on the fitted out-of-fold predictions and omit uncertainty from retraining the overlapping folds; they are not full external-validation uncertainty estimates.", "",
             "## Results", "", "| C | Model | Log loss | Normalized RPS | Accuracy |", "|---|---|---:|---:|---:|"]
    for result in summaries:
        lines.append(f"| {result['C']:g} | {result['model']} | {result['mean_log_loss']:.4f} | {result['mean_normalized_RPS']:.4f} | {result['accuracy']:.3f} |")
    lines += ["", "Primary C=1 paired comparisons (positive values favor the expanded model):", ""]
    for contrast in contrasts:
        if contrast["primary"]:
            lo, hi = contrast["bootstrap_95_percentile_interval"]
            lines.append(f"- {contrast['comparison']}, {contrast['metric']}: improvement {contrast['mean_improvement']:.4f}; conditional 95% interval [{lo:.4f}, {hi:.4f}]; {contrast['participants_improved']}/68 participant averages improved.")
    lines += ["", f"All {len(fit_diagnostics)} fits completed. Total convergence warnings: {sum(x['convergence_warnings'] for x in summaries)}. Maximum iterations used: {max(x['max_iterations'] for x in summaries)}.", "", "## Interpretation limits", ""]
    lines += [f"- {item}" for item in assumptions]
    lines += ["- Prediction from past ratings can reflect stable individual response tendencies, response consistency, and shared case order as well as changing beliefs. It does not isolate a feedback-driven psychological updating mechanism.",
              "- At the first forecast trial, history proportions and latest-rating indicators coincide. Ridge penalties on correlated and redundant representations can influence incremental model comparisons; regularization sensitivity should accompany the primary result.",
              "- The 50 missing decisions were not filled in or recoded. This rating analysis does not model decisions, so they do not remove complete direct ratings from the prediction dataset.", ""]
    (args.output_dir / "corrected_prediction_methods_results.md").write_text("\n".join(lines))
    print(f"Saved results in {args.output_dir}; elapsed {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()

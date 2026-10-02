# Dynamic trust in clinical AI

Research code accompanying **A conceptual and mathematical framework for dynamic
trust in clinical AI** by Avishek Choudhury and Yeganeh Shahsavar.

The repository supports the mathematical examples, the auxiliary analysis of
repeated recommendation-trust ratings, and numerical verification. The
mathematical appendix (S1) and empirical appendix (S2) remain separate manuscript
supplements. Code is documented here as a repository rather than an S3 supplement.

**This repository contains code only.** It contains no participant records,
participant-level predictions, previously generated results, manuscript drafts,
or physiology exports. The clinical inputs and their access conditions are
documented separately below. No archival DOI or software reuse license has been
assigned. The code does not establish psychological or clinical validation of
the proposed framework.

## Quick start without clinical data

The source analysis environment used Python **3.12.13**. Direct package versions
are pinned in `requirements.txt`; transitive dependencies may resolve differently
on other platforms.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/smoke_check.py
python work/make_figure1.py
python work/make_figure2.py
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in
PowerShell. Run the commands below from the repository root. Keep the existing
folder structure because several scripts infer the root from their location.

`smoke_check.py` checks hypothetical capability/prediction examples, probability
normalization, an analytic derivative, numerical integration, and the exclusion
of current/future ratings from forecasts, using synthetic inputs only. Passing
these checks establishes limited implementation checks, not empirical adequacy.
`make_figure1.py` generates the current conceptual framework as
`work/framework_figure.png` and `work/framework_figure.svg`.
`make_figure2.py` generates `work/figure2.png` and `work/figure2.svg`. Both scripts
work without clinical data and are separate from manuscript editing.

## Clinical inputs

To reproduce the empirical analysis, access to the authorized original inputs
is required. Place the files locally in `upload/` after obtaining them through
an approved route:

- `Corrected_Clinical_Trial_Long_Data.csv`
- `Appendix_A_Actual_Sequence.csv`

The script names are retained for compatibility with the original analysis.
The [data dictionary](docs/data_dictionary.md) explains the required columns,
fixed study dimensions, timing, and excluded fields. The earlier `Clean_R3.csv`
is not an interchangeable source for this analysis. The other historical EEG,
GSR, feature, and participant-flow exports are not dependencies of these scripts.

```bash
python scripts/validate_inputs.py
```

The check validates participant/trial keys, category ranges, fixed sample
structure, and agreement with the sequence key. It does not adjudicate the
clinical correctness of case labels. Missing decisions remain missing; the
rating forecasts use complete ratings and do not fit decisions.

## Reproduce estimation and baseline forecasts

These steps must finish before the subsequent full-history calculations.
Bootstraps, recovery simulations, and leave-one-person-out fits can take substantial
computation time. No previously fitted clinical outputs are bundled.

```bash
python work/corrected/predict_corrected.py --input upload/Corrected_Clinical_Trial_Long_Data.csv --sequence upload/Appendix_A_Actual_Sequence.csv --output-dir work/corrected/prediction --bootstrap 10000
python work/corrected/ordinal_dependence.py --input upload/Corrected_Clinical_Trial_Long_Data.csv --output work/corrected/dependence --bootstraps 200 --simulation-replicates 100
python work/corrected/ordinal_dependence_prediction.py --input upload/Corrected_Clinical_Trial_Long_Data.csv --output work/corrected/dependence
```

The first command fits categorical multinomial forecasts with fixed primary
regularization C=1 and sensitivity values C=0.1 and C=10. The second fits static
and autoregressive Gaussian ordinal dependence by all-pairs composite likelihood,
refits 200 stratified participant bootstrap datasets, and generates 100 recovery
datasets at each of three persistence settings. The third estimates population
parameters separately in each leave-one-person-out fold and saves one-lag
forecasts and fits needed by the full-history filter.

The original baseline forecast intervals and generated baseline report are
intermediate outputs. Use the common stratified score comparisons from
`revised_scores.py` below for final reporting. A centered baseline composite is
not a previous-trial response. Category probabilities are forecasts of an observed
rating; they are not estimates of a clinician's calibrated capability belief.

## Model checks and final full-history forecasts

```bash
python work/peer_review/analysis/model_checks.py
python work/peer_review/analysis/full_history.py --person-nodes 481 --residual-nodes 64
python work/peer_review/analysis/revised_scores.py
```

`model_checks.py` generates 2,000 fitted-model datasets, calculates 10,000
stratified participant resamples of observed summaries, and examines three margin
pseudocounts. Simulation ranges condition on fitted parameters and are not
confidence intervals or calibrated goodness-of-fit p values.

The full-history filter uses all completed preceding ratings in both the static
and dynamic models. Each current forecast is saved before the current rating
updates the filtering distribution. The held-out person's ratings do not enter
population fitting or case-margin estimation.

`revised_scores.py` recomputes category-probability scores and uses the same 10,000
participant resamples, stratified by series, for paired comparisons. Its
`all_forecast_scores.csv`, `all_paired_score_contrasts.csv`, and
`revised_scores.json` are the final comparison outputs. Intervals condition on
saved out-of-fold predictions and omit uncertainty from complete retraining and
model development. Do not interpret them as external-validation intervals.

## Numerical verification

The final reporting grid is 481 person nodes × 64 residual nodes. Generate the
comparison grids before running the grid/prefix checks:

```bash
python work/peer_review/analysis/full_history.py --person-nodes 61 --residual-nodes 32
python work/peer_review/analysis/full_history.py --person-nodes 121 --residual-nodes 64
python work/peer_review/analysis/full_history.py --person-nodes 241 --residual-nodes 64
python work/peer_review/analysis/full_history.py --person-nodes 241 --residual-nodes 96 --output work/peer_review/verification
python work/peer_review/analysis/check_full_history.py
python work/peer_review/verify_full_history.py
python work/final_verification/verify_statistics_outputs.py
python work/final_verification/verify_statistics_integrations.py
```

The smaller grids are numerical diagnostics, not alternative primary results.
`check_full_history.py` compares the final grids, checks an independently coded
static filter, and perturbs target/future ratings to check forecasting order.
`verify_full_history.py` adds adaptive static integration and early-prefix
multivariate-normal rectangle comparisons. The final two scripts recompute
scores, paired intervals, source counts, model-check summaries, and independently
coded integration calculations from the regenerated outputs. Stochastic
multivariate-normal CDF comparisons may vary slightly across platforms.

## Empirical figures

After all primary analyses above finish:

```bash
python work/peer_review/make_empirical_figures.py
```

This generates Figure 3 and the recovery figure in S2 (`figure_s2_1`) in PNG, SVG,
and PDF. Review regenerated values and captions together before replacing
manuscript figures. This repository does not rebuild or edit the manuscript.

## Files and reproducibility limits

| Location | Purpose |
| --- | --- |
| `work/corrected/` | Original estimation and baseline prediction implementations |
| `work/peer_review/analysis/` | Full-history forecasts, model checks, final scores |
| `work/peer_review/verify_full_history.py` | Additional numerical comparisons |
| `work/final_verification/` | Independent score and integration calculations |
| `work/make_figure1.py` | Current conceptual framework figure, without document editing |
| `work/make_figure2.py` | Data-free mathematical figure |
| `scripts/` | Input validation and synthetic smoke checks |
| `docs/data_dictionary.md` | Required schemas and interpretation |
| `upload/README.md` | Local input placement; no data included |

The folder name `peer_review` is a historical implementation path, not a claim
that these files constitute external journal review. Its earlier automated review
reports are excluded. The source archive's eight-file audit and document-building
scripts are also excluded because they are not required for this two-input
clinical pipeline.

The stable-person variance fraction is **η** in the paper and `omega` in original
numeric fields; it is unrelated to the theoretical evidence weight **ωₜ**.
Parameter recovery from a model's own simulations does not demonstrate that the
model adequately represents observed ratings. This analysis does not identify
capability-probability units, a causal feedback gain, elapsed-time forgetting, or
physiological trust biomarkers. Forecasts concern new people following the same
two fixed case sequences.

## Release and citation

Repository: [Dynamic trust in clinical AI](https://github.com/avishekadamas-glitch/dynamic-trust-clinical-ai).

For reproducible citation, identify the exact release tag or commit used. The
manuscript's Code availability statement should cite this repository and that
version. No archival DOI or software reuse license has been assigned. Data
access is separate from code availability.

Associated manuscript: Avishek Choudhury and Yeganeh Shahsavar, *A conceptual and
mathematical framework for dynamic trust in clinical AI*. Add the article's
bibliographic details when published.

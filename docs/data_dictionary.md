# Input schema and analysis scope

The schemas below document the two original clinical files. This repository
contains no rows from either file. Column descriptions reflect their use in the
analysis; they are not a replacement for the study instrument or data codebook.

## `Corrected_Clinical_Trial_Long_Data.csv`

One row represents one participant at one chronological trial. The implementation
is specific to 68 people, 34 in each of two fixed case series, with 21 ratings each
(1,428 rows). It is not a general-purpose data loader for other study designs.

| Column used | Meaning and required coding | Use |
| --- | --- | --- |
| `ID` | Unique participant code; stable across the person's trials | Grouping, leave-one-person-out fitting, bootstrap unit |
| `CaseSeries` | Fixed sequence A or B | Group-specific margins and design cells |
| `Trial` | Actual chronological position, integer 1–21 | Ordering and forecast target |
| `RawTaskNumber` | Original task identifier | Reconciliation with the sequence key; not substituted for chronology |
| `direct_trust_item` | Observed recommendation-trust rating, integer 1–7 | Ordinal/categorical endpoint; not a calibrated probability |
| `Accuracy` | Supplied correctness label, `Correct` or `Incorrect` | Sequence reconciliation and descriptive checks; excluded from forecast predictors |
| `decision_accept` | Observed acceptance 1, rejection 0, missing left missing | Descriptive verification only; not used to fit rating forecasts |
| `Job_Category` | Supplied professional category | Descriptive sample verification only |

The core estimation and prediction require `ID`, `CaseSeries`, `Trial`,
`RawTaskNumber`, `Accuracy`, and `direct_trust_item`. The final descriptive-output
verification additionally reads `decision_accept` and `Job_Category`.

The original header also contains the following columns, which are not needed by
the repository's rating models: `transparency`, `likelihood_to_act`,
`trust_composite`, `difficulty_ordinal`, `Difficulty`, `CaseItem`,
`baseline_trust_composite`, `intent_to_use_AI`, `self_confidence`, `exp_binary`,
`Trial_c`, `prior_trust_c`, `intent_c`, `selfconf_c`, `difficulty_c`,
`decision_recode`. Their presence does not authorize interpreting them as prior
ratings or replacing missing decisions with recoded values.

The data-free checks use locally generated synthetic arrays that are never saved
as clinical files. They must not be described as observations or evidence of
clinical validation.

## `Appendix_A_Actual_Sequence.csv`

One row represents one series/trial design cell, for 42 unique cells.

| Column | Meaning and use |
| --- | --- |
| `CaseSeries` | A or B; joins the clinical file |
| `Trial` | Chronological position 1–21; joins the clinical file |
| `RawTaskNumber` | Checked for exact agreement with the clinical file |
| `Accuracy` | Supplied correctness label; checked for exact agreement |
| `Patient_profile` | Case text in the original header; not required by any included computation |
| `AI_diagnosis_and_recommendation` | Recommendation text in the original header; not required by any included computation |

The sequence file supplies the design mapping. Agreement between two files does
not independently establish clinical correctness of their labels.

## Timing and notation

Ratings were collected before correctness feedback. Choice and rating appeared
on the same screen; actual choice/rating order was not verified by timestamps.
These analyses predict subsequent ratings and do not use a current rating to
explain a concurrent decision.

The empirical stable-person variance fraction is written as **η** in the paper
and `omega` in the original code. This implementation label does not refer to the
framework's theoretical evidence weight **ωₜ**. The autoregressive coefficient
`phi` is trial-indexed; it is not an estimated elapsed-time forgetting rate.

## Access and generated files

The repository does not provide public access to the clinical data or imply
data-sharing approval. The full empirical analysis requires the authorized inputs
described above. Without those inputs, users can run the mathematical examples
and synthetic implementation checks. Generated predictions and fit files are
ignored by Git.

# Methods represented by the release

This release separates preserved scientific routines from portable adapters. The
network forward passes, log-variance heads, report-key error generator, radio
regeneration, Core cell derivations, Gaussian-NLL training loop and coordinate GB
selection come from the implementations used for the manuscript. The command-line
interface, prepared-cache format and explicit joins are release adapters. Source
file hashes and adaptations are listed in [source_provenance.json](source_provenance.json).
The adapters were tested on synthetic inputs; packaging did not rerun the study.

## Input and target contract

Each eligible input contains ten retained reports at exactly `t-9, ..., t` seconds.
The target is the raw, unfiltered scene coordinate at exactly `t+5`. A gap after the
input endpoint does not change the horizon. `data.exact_windows` retains unlabeled
opportunities and marks label availability rather than shifting targets. Model
inference does not read target coordinates; evaluation uses only available labels.

`feature_engineering` retains the historical row and vehicle filters and the
non-coordinate feature derivations. Serving-cell assignment must exist and scalar
speed must be at least 0.1 m/s. Vehicle eligibility is then computed on retained
rows: duration at least 60 s, mean speed above 1 m/s, and fraction above 0.5 m/s
greater than 0.4. The release uses fixed vehicle identities, including the original
ten demonstration vehicles assigned to test; it does not redraw a split.

Feature order is executable in `data.RAN_COLS`, `fusion.CORE_COLS`, and `data.ALL_COLS`:

- RAN: six received powers, six SINRs, strongest-report power/SINR, six serving-cell
  indicators, six changes for each radio quantity, power range, SINR range and
  strongest-minus-second power margin (35 values).
- Core: six cell indicators, two fixed cell-centroid coordinates, supplied scalar
  speed, transition flag, time since transition, two unnormalized transition
  direction components, dwell time and scalar-speed change (15 values).
- Centralized: the union of those two sets, counting shared indicators once (44).
  The historical no-speed comparator removes speed and its change (42).

No predictor receives observed UE coordinates or coordinate-derived motion.
The fixed `BSXY` in `cell_state.py` preserves the centroid values used in trained
inputs; these constants are not recomputed from publication figure geometry.
Radio changes are clipped to ±30 dB and speed changes to ±10 m/s. They are
report-to-report differences, even across a gap. Time since the latest transition
is zero before the first transition and clipped at 300 s. Dwell is clipped at
600 s. Direction components independently retain their most recent nonzero value.

## Preprocessing and radio augmentation

The cached augmentation is deterministic by vehicle, timestamp, primitive and
observation seed (2026100900); overlapping windows share the same perturbation.
At vehicle level, 30% receive no radio corruption. Otherwise the Gaussian scale
is uniform on [0,6] dB, missing probability on [0,0.2], and power quantization is
selected from 0,1,2,5 dB. SINR is not quantized. Missing primitive fields are filled
using means of **unique clean reports from fitting vehicles only**, and dependent
radio features are then regenerated. Serving-cell indicators and all Core fields,
including speed, remain fixed. When powers tie, the contemporaneous cell proxy
breaks the tie if it is among the strongest reported cells.

Each scaler is fitted to the flattened fitting windows, so a report that occurs
in several fitting windows contributes repeatedly to scaling. The clean-domain
scaler is reused in the augmented phase. The centralized models fit their scaler
to augmented fitting windows. No held-out or model-selection windows fit a scaler.

## Domain networks and loss

The RAN encoder projects 35 inputs to 128 dimensions, adds sinusoidal positions,
applies LayerNorm and four pre-norm Transformer layers (eight heads, 512-unit
feed-forward layer, GELU, dropout 0.1). The last token feeds the position and
log-variance heads. The position head is LayerNorm → 128-to-64 → GELU → dropout →
64-to-2. The variance head is 128-to-32 → ReLU → 32-to-2.

The Core encoder is a two-layer bidirectional LSTM with hidden size 128 and
inter-layer dropout 0.1. It reads `lstm_out[:, -1, :]`; this is not a concatenation
of the final hidden state from both directions. Its position head is 256-to-128 →
BatchNorm → ReLU → dropout → 128-to-64 → BatchNorm → ReLU → 64-to-2. The variance
head is 256-to-64 → ReLU → 64-to-2.

Both variance heads initialize the final bias to 4.6 and return
`exp(clamp(log_variance, -4, 10))`. Values remain learned, uncalibrated variances.
The mean Gaussian NLL is the mean over both axes and windows, including the
Gaussian one-half factor and omitting the constant. Equivalently its summed
log-variance and standardized-square-error terms have factor `1/(4N)`.

## Training and output stacking

The domains train independently. AdamW uses batch size 512, weight decay 1e-5 and
gradient clipping at 1. The clean phase has learning rate 5e-4, a 40-epoch cosine
schedule to 1e-6, and at most 40 epochs. Patience is ten epochs with 0.001 minimum
improvement for resetting patience; every strict validation-loss minimum is still
eligible as the selected checkpoint. The selected clean model initializes the
augmented phase at learning rate 1e-4, with eight RAN epochs or five Core epochs
and a per-batch cosine schedule. The pre-fine-tuning checkpoint is eligible too.

Vehicle folds, internal development vehicles and final model-selection vehicles
are fixed in `configs/vehicle_roles.json`. Each outer held-out fold is excluded
from imputation, scaling, fitting and checkpoint selection. Five-fold outputs
cover every scored training window exactly once. GB fits use **augmented OOF**
outputs; their coordinate snapshots are selected using **augmented full-training**
outputs on model-selection vehicles. Primary test radio is clean. Neither the
fold models nor the five independently fitted seeds are ensembled.

## GB interfaces and selected snapshots

The output vector is `[RAN x, RAN y, RAN var_x, RAN var_y, Core x, Core y,
Core var_x, Core var_y]`. Separate histogram-based GB regressors predict the two
absolute coordinates; the fused output has no variance head.

| Name | Inputs | Count |
|---|---|---:|
| `fusion_full` | Both position and variance pairs | 8 |
| `fusion_means` | Both position pairs | 4 |
| `fusion_rvar` | Both position pairs and RAN variances | 6 |
| `fusion_cvar` | Both position pairs and Core variances | 6 |
| `ran_post` | RAN position and variances | 4 |
| `ran_speed` | RAN outputs plus ten speed/change pairs | 24 |
| `ran_core_raw` | RAN outputs plus ten Core15 reports | 154 |

All coordinate fits use learning rate 0.1 and at most 63 leaves, with internal
sklearn early stopping disabled. Fit in 100-tree increments, stop after five
increments without lower model-selection MSE or at 12,000 trees, and retain the
minimum-MSE snapshot, which need not be the final fitted ensemble. The API's
smaller budget override exists for tiny tests; the CLI uses the manuscript budget.

Some seed-42 fits predate the common 12,000 cap. Converged fits were reused; capped
fits were extended from selected warm-start snapshots with overlapping validation
steps checked. `fit-fusion --resume-selected` preserves that continuation logic.
A fresh fit is a method implementation, not a claim to replay the exact historical
sequence of reuse and extension. See `configs/fitting_history.json`.

## Centralized and sensitivity comparisons

The all-features and no-speed references are separate seed-42 BiLSTMs, each with
one 40-epoch-maximum augmented-radio phase and coordinate MSE. Their position head
has a dropout before its first linear layer and is distinct from the Core head.
The no-speed row is an earlier comparator; these are not five-seed primary models.

`cell_state.retention_core` preserves the last valid association at floor reports;
without a prior valid cell the state is unknown and cell-derived fields are zero.
Unknown-to-known attachment is not a transition. Only the Core cache changes;
RAN models/outputs and all roles are reused. The published sensitivity used 25 new
Core fold fits, four new full Core fits plus the existing seed-42 full Core model,
and 40 coordinate GB fits for four variance configurations. The primary stays fixed.
Use the original float64 report mask, not a newly inferred mask from rounded input.

`speed.generate` implements the precision-corrected source stream. Add keyed
Gaussian error/bias in float64, clamp speed to nonnegative, align arrivals, compute
the adjacent received-speed change in float64, clip to ±10, then cast both fields
to float32. All evaluated models stay fixed after nominal-speed training. Small
noise used three seeds 2026101601–03 and two scales 0.141/0.205 m/s on the full
primary test set. Larger noise/bias/delay used three different realization seeds
2026101301–03 and a common exact-source-availability cohort: 233,711 opportunities,
of which 220,918 have labels from 4,010 vehicles. Availability uses no model error.
`speed.common_exact_cohort` reproduces its saved opportunity IDs. These sensitivity
primitives and saved-output verification are provided; a one-command replay of
all study perturbation inference is not claimed without source streams/checkpoints.

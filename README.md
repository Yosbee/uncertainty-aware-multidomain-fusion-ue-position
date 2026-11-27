# Output-Level RAN–Core Cooperation for Future UE Position Prediction

Research code for independently trained RAN and Core predictors that exchange
predicted positions and learned per-axis variances with a separate fusion model.
The primary task predicts a UE's position five seconds ahead from ten consecutive
reports, without observed UE position histories as inputs.

This source revision matches the current manuscript's RAN35/Core15 comparison.
The manuscript snapshot is identified in
[`metadata/manuscript_version.json`](metadata/manuscript_version.json).

## Two ways to use the repository

| Purpose | Entry point | Additional inputs |
| --- | --- | --- |
| Fit domain predictors and output fusion | `python3 -m uefusion` | Prepared reports, raw future-coordinate labels, vehicle roles |
| Predict with fitted models | `python3 -m uefusion infer` / `predict-fusion` | Prepared inputs and checkpoint/scaler bundles |
| Recompute paper results | `analysis/verify_results.py` | External saved predictions, targets, identities and analysis masks |
| Recompute the original confidence intervals, ranked groups and variance/bypass analyses | `analysis/primary_archive_score.py` | Extracted primary numerical archive |

Large input arrays, prediction archives and checkpoints are not stored in this
checkout. See [`data/README.md`](data/README.md) for their scope and
[`docs/VERIFICATION.md`](docs/VERIFICATION.md) for the exact numerical-verification
inputs. The code and data are separate deliverables; no public data-download URL
has been assigned by this repository preparation.

## Installation and checks

Use Python 3.10 or later. A virtual environment is recommended.

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-tested.txt
python3 -m pip install --no-deps -e .
python3 -m uefusion --help
python3 -m pytest -q
```

`requirements-tested.txt` records the environment used for the portable-code
checks. The tests use synthetic inputs to check time matching, role separation,
model outputs and evaluation logic; they do not repeat the study's training runs.
For saved-result verification alone, the analysis programs use NumPy and do not
require PyTorch or a GPU.

## Recompute saved primary results

After obtaining the primary numerical archive separately:

```sh
python3 analysis/verify_results.py \
  --mode primary \
  --primary external_data/primary_numerical_reproduction.zip \
  --out outputs/primary
```

The command checks sample counts and identities, computes coordinate errors,
and writes per-seed and aggregate CSVs. It uses each seed's own percentile and
then averages across seeds. It does not ensemble predictions. The floor,
retention and speed-sensitivity modes additionally require their saved masks
and contracts; examples are in [`docs/VERIFICATION.md`](docs/VERIFICATION.md).

## Method and fitting

The RAN predictor is a Transformer and the Core predictor is a bidirectional
LSTM. Each produces two coordinates and two variances. Their models are fitted
separately with Gaussian negative log-likelihood. Vehicle-disjoint out-of-fold
outputs train coordinate-wise histogram gradient-boosting regressors; snapshots
are selected using model-selection vehicles and full-training domain outputs.

The fusion implementations cover positions only, RAN variances only, Core
variances only and both variance pairs. Comparators include a RAN postprocessor,
a RAN + speed postprocessor, a Core-feature reference and the seed-42 centralized
models. The supplied speed is an assumed application input in the nominal
comparison. Cell-retention and speed-error sensitivity routines are separate
from that primary protocol.

[`docs/TRAINING.md`](docs/TRAINING.md) gives the input schema, runnable command
sequence and the historical fitting conditions.
[`docs/METHODS.md`](docs/METHODS.md) maps modules to the scientific method.
[`docs/source_provenance.json`](docs/source_provenance.json) records the origins
of retained routines and the portable adapters. Full study training has not been
repeated as part of this code reorganization; saved-result verification is
reported separately.

## Layout

```text
src/uefusion/     Input processing, models, training, OOF and fusion
configs/         Study settings, vehicle roles and sensitivity conditions
analysis/        Saved-result metrics and numerical verification
metadata/        Evidence hashes, expected results and manuscript snapshot
tests/           Synthetic contract, parity and calculation checks
docs/            Method, fitting, verification and scope documentation
data/            External-data instructions (no research arrays)
```

See [`docs/SCOPE.md`](docs/SCOPE.md) for the repository's code and artifact scope. Source code is covered by [`LICENSE`](LICENSE);
external data has separate terms described in [`DATA_LICENSE`](DATA_LICENSE).
Author and citation metadata is supplied in [`CITATION.cff`](CITATION.cff).

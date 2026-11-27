# Repository scope

This source release corresponds to the manuscript, **Output-Level RAN-Core
Cooperation for Future UE Position Prediction**.

## Code and evidence are separate deliverables

The source tree contains input preparation, the two independently trained domain
models, out-of-fold stacking, the seven gradient-boosting comparisons, centralized
references, and numerical analysis of saved predictions. Training and inference
require the prepared input data and, for inference, fitted checkpoints. These
large artifacts are not embedded in this Git revision.

The numerical verification path reads external saved predictions, targets,
vehicle/time identifiers and analysis masks. It recomputes reported metrics
without training or running a prediction model. `docs/VERIFICATION.md` describes
the required files and commands; `metadata/` records evidence hashes and expected
aggregate results. A repository checkout by itself is not the prediction archive.

The portable interfaces use explicit paths and retain the scientific routines
identified in `docs/source_provenance.json`. Adapters replace the original
machine-specific orchestration. Their unit and saved-result checks do not amount
to repeating the full training campaign. The historical fitting lineage,
including reused fits and finite fitting budgets, is described in
`docs/TRAINING.md`.

## Included method

- Ten consecutive one-second reports and a raw-coordinate target at exactly
  five seconds after the input endpoint.
- RAN: 35 features; Core: 15 features, including the supplied scalar speed and
  its report-to-report change. Observed UE position histories are excluded.
- Separate position/variance heads and domain training; vehicle-disjoint folds
  for out-of-fold outputs, with separate model-selection vehicles.
- Four variance-input configurations, a RAN postprocessor, a RAN + speed
  postprocessor, and a Core-feature reference.
- Seed-42 centralized comparators, with their own fitting conditions retained.
- Primary scoring, power-floor strata, cell-retention sensitivity, speed-error
  sensitivity, and the original bootstrap/variance/bypass calculation path.

## Excluded from the current checkout

Historical observed-position models and experimental branches unrelated to the
current comparison; manuscript editing scripts and internal review reports;
papers and submission PDFs; raw radio maps and mobility files; trained model
binaries; per-window prediction archives; local caches and execution logs.

The release history starts from this manuscript-aligned source tree. Earlier
development history is archived separately and is not included in this
repository's branches or tags.

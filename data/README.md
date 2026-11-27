# External data

Keep downloaded or locally prepared data outside Git, for example in
`external_data/`. The directory is ignored by Git.

There are two independent entry points:

1. **Model fitting and inference:** retained engineered reports, the unfiltered
   coordinate table used for exact future targets, and vehicle role metadata.
   See `docs/TRAINING.md` for the schema and preparation commands. Fitted
   checkpoint/scaler bundles are additionally required to predict without
   retraining.
2. **Numerical verification:** saved coordinate predictions and targets, shared
   identifiers, floor masks, and sensitivity-condition contracts. See
   `docs/VERIFICATION.md` and `metadata/evidence_files.json`.

No prediction archive, original mobility trace, radio map, or trained checkpoint
is included in this revision. Aggregate CSVs and hashes describe the evidence;
they are not substitutes for the per-window inputs used by verification.

No public archive URL or DOI is assigned by this repository preparation step.
The evidence distribution location will be recorded when an actual archive has
been released. Refer to `DATA_LICENSE` for the distinction between the software
license and data usage terms.

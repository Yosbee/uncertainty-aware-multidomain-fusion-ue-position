# Saved-result verification

These CPU programs evaluate saved coordinates. They do not train a model, run
inference, generate noise or regenerate the radio map. NumPy is the only external
Python dependency. Prediction archives and model binaries are not stored in Git.

## Evidence layout

Supply the external evidence files described by `metadata/evidence_files.json`.
The manifest records byte sizes and SHA-256 hashes of locally verified source
files. It is not a claim that an external data release has already been published.
Release URLs and any repackaged file hashes must be assigned before public release.

```
evidence/
  primary/
    primary_numerical_reproduction.zip
    window_strata.npz
  retention/
    seed42/predictions.npz
    ... seed46/predictions.npz
  speed/
    small/
      scoring_contract.npz
      seed42/nominal_r0/predictions.npz
      seed42/noise_0.205_r2026101601/predictions.npz
      ...
    large/
      scoring_contract.npz
      seed42/nominal_r0/predictions.npz
      seed42/noise_2_r2026101301/predictions.npz
      ...
```

The primary ZIP may instead be extracted. Its historical directory prefix is
accepted automatically; paths are resolved by a unique file name, without
executing code from the archive. `--retention`, `--small-speed` and `--large-speed`
each point at the directory containing the seed directories. The speed directories
also contain their own scoring contract. Large-stress inputs must be the corrected
release in the manifest, not superseded outputs.

```sh
python3 analysis/check_evidence.py --manifest metadata/evidence_files.json --data-root evidence
python3 analysis/verify_results.py --mode all \
  --primary evidence/primary/primary_numerical_reproduction.zip \
  --floor evidence/primary/window_strata.npz \
  --retention evidence/retention \
  --small-speed evidence/speed/small \
  --large-speed evidence/speed/large \
  --expected metadata/expected_results.csv --out results/verification
```

Use `--mode primary` for aggregate primary results (only `--primary` required),
`--mode floor` for the primary floor strata, `--mode retention` for retention,
or `--mode speed` for either/both speed archives. Stratified modes require
`--floor`. A speed mode never silently omits a missing seed, condition or realization.
An archive-wide run is the documented expected-table check. With primary mode
alone, omit `--expected`, because that file also contains primary stratum rows.

## Identity and cohort checks

The primary contract contains 286,657 opportunities, of which 272,201 have labels,
from 4,032 test vehicles. The horizon is exactly five seconds. A scored sample ID,
vehicle ID, window endpoint, target timestamp and two-coordinate truth define the
identity. Floor labels and retention records are joined by scored sample ID and
checked against all supplied identity fields. Speed predictions are first joined by
opportunity ID to their own contract; labelled records are then checked by scored
sample ID, vehicle, endpoint, target time and truth against the primary contract.
Duplicate IDs, missing IDs and inconsistent targets are errors.

Primary seed arrays have a shared identity file, rather than a separate identity
inside each seed file. File hashes therefore protect this original row-order
contract. The verifier does not claim to recover row identity from coordinates.

Small-speed results use all 272,201 labelled primary samples. Large stresses use
**only the exact-source-time mask: 220,918 windows from 4,010 vehicles**. The
249,900 labelled available windows in the large-speed contract are not the paper's
main large-stress cohort. Every stress condition is evaluated with its own
same-cohort nominal baseline; RAN outputs must remain unchanged across conditions.

## Metrics and aggregation

- Euclidean position error is recomputed from saved predicted and true coordinates,
  using float64 arithmetic; stored prediction precision is preserved.
- Mean, P95 and coordinate-vector squared error are computed per realization.
  P95 uses NumPy's linear quantile interpolation. MSE is the mean squared Euclidean
  error, not the average over coordinate elements used in a coordinate loss.
- Realization metrics are averaged within each training seed, then the five seed
  metrics are averaged. Percentiles are not taken after pooling all seeds, and
  predictions are not ensembled. Centralized references have one seed only.
- A 50 m failure has error strictly above 50 m. Rescues and new harms use each
  seed's corresponding RAN prediction. Final failures = RAN failures − rescues +
  new harms. Both the mean of seed percentage reductions and the percentage from
  mean counts are exported, with different column names.
- `partial_floor` is the publication label for the historical `mixed` key.
  Strata are recomputed from counts of floor reports in the ten-report input.
  All-floor windows are additionally divided by prior valid cell availability.
- A stratum's mean-error contribution is its within-stratum error difference
  multiplied by its share of the cohort. Both mean seed shares and share of the
  overall mean reduction are exported, because the two quantities differ.
- The four fusion configurations are `fusion_means` (positions only),
  `fusion_rvar`, `fusion_cvar` and `fusion_full`. `ran_core_raw` is the Core-feature
  reference; `Central44` and `Central42` are the seed-42 centralized predictors
  with and without the two speed inputs, respectively.

`per_realization.csv` retains every seed and realization. `per_seed.csv` averages
realizations only. `summary.csv` adds seed SDs. Contribution tables preserve
negative values and adverse seeds. The committed expected table was copied from
previously recorded result tables; it is not generated from the verifier's output.
The expected check uses absolute tolerance 1e-7 and relative tolerance 1e-9.

## Bootstrap, tails and variance-based selection

The original numerical-archive scorer is retained as a separate portable entry:

```sh
OPENBLAS_NUM_THREADS=1 python3 analysis/primary_archive_score.py \
  --primary evidence/primary/primary_numerical_reproduction.zip \
  --out results/primary_archive
```

It verifies the archive manifest, then recomputes the primary and centralized
metrics, fixed error-ranked tails, validation-defined RAN-variance groups,
variance-based bypass and **conditional paired vehicle-bootstrap intervals**.
It uses the archive's 1,000 preserved shared vehicle multiplicities; it does not
sample new bootstrap draws. The reported intervals condition on the trained
models and use vehicle clusters. It checks 11 tolerance thresholds and the
archive's own expected tables. Its scope is the original primary numerical
comparison; it does not cover later floor strata or retention/speed sensitivity.

The two programs therefore have complementary scopes. Neither independently
reconstructs radio-floor masks from the ray-tracing map, reruns training or
validates raw simulation geometry. The result verifier accepts the saved floor
labels; those labels' data-generation provenance is documented separately.

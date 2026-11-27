# Running the research implementation

The submission's result-verification archive contains saved model outputs; it is
not a training dataset. Training requires retained engineered report rows and
unfiltered future coordinate labels in the schema below. They are separate from
the saved-prediction reproduction path. No training, inference or noise generation
runs merely by importing the package or displaying help.

Use `python3 -m uefusion --help` after installing the package. Commands refuse to
overwrite existing runs. Pickle/checkpoint inputs must be your own trusted files.

## Required inputs

`reports.csv` has `vehicle_id` (string), `timestamp` (integer seconds) and the 44
columns in `uefusion.data.ALL_COLS`. They are retained, chronologically ordered
engineered reports, including all earlier retained records needed for causal
changes and cell state. `feature_engineering` contains historical filtering and
derivation functions for matching source schema. The original scalar-speed field
must be supplied; it is never derived from UE coordinates.

`raw_targets.csv` has `vehicle_id`, `timestamp`, `scene_x`, `scene_y`, before the
report speed/association filters. The target coordinate at t+5 remains eligible
when intermediate future observations were filtered out. Missing labels are
marked, not interpolated or replaced by an offset row. The frozen
`configs/vehicle_roles.json` includes both original train/validation/test source
splits and the fit/development/held-out roles used for training.

```sh
python3 -m uefusion prepare --reports /data/reports.csv \
  --targets /data/raw_targets.csv --splits configs/vehicle_roles.json \
  --output /data/prepared
```

This command constructs report-level clean/augmented arrays and exact-time window
lookups. It generates the prescribed radio augmentation **only when invoked**.
`clean_reports.npy`, `augmented_unimputed_reports.npy`, `vehicle_ids.npy`,
`timestamps.npy`, `feature_columns.json`, and `windows/{train,val,test}.npz` make up
the portable cache. Window files hold report indices, IDs/times, targets and label
availability. Original study cache formats require an explicit conversion; an
arbitrary existing folder is not silently interpreted as this schema.

## Domain fits and held-out outputs

Run one model, seed and fold at a time, for seeds 42–46 and folds 0–4/full:

```sh
python3 -m uefusion train --dataset /data/prepared \
  --splits configs/vehicle_roles.json --model ran --seed 42 --fold 0 \
  --output runs/seed42/fold0/ran --device cuda:0
python3 -m uefusion infer --dataset /data/prepared \
  --checkpoint-dir runs/seed42/fold0/ran --split train --condition augmented \
  --output outputs/seed42/ran_fold0.npz --device cuda:0
```

Use `--model core` for the Core predictor. A fold prediction contains only its
held-out vehicles. `--fold full` fits all designated training vehicles and selects
using model-selection vehicles. Its augmented validation outputs are produced
with `infer --split val --condition augmented`; its test outputs use
`infer --split test --condition clean`. Inference keeps all past-eligible windows
and does not read target coordinates. The `train` command does not open test
windows or test labels.

```sh
python3 -m uefusion assemble-oof --dataset /data/prepared \
  --outputs outputs/seed42/ran_fold0.npz outputs/seed42/ran_fold1.npz \
  outputs/seed42/ran_fold2.npz outputs/seed42/ran_fold3.npz \
  outputs/seed42/ran_fold4.npz --output outputs/seed42/ran_oof.npz
```

Repeat assembly for Core. Every scored training window must have one prediction
from a model whose fit and checkpoint-selection vehicles exclude that window's
vehicle. Joining validates window ID, vehicle ID, input time and target time.

## Fusion fitting and inference

```sh
python3 -m uefusion fusion-inputs --dataset /data/prepared --split train \
  --ran outputs/seed42/ran_oof.npz --core outputs/seed42/core_oof.npz \
  --output inputs/seed42/train.npz
python3 -m uefusion fusion-inputs --dataset /data/prepared --split val \
  --ran outputs/seed42/ran_val_augmented.npz \
  --core outputs/seed42/core_val_augmented.npz --output inputs/seed42/val.npz
python3 -m uefusion fit-fusion --train inputs/seed42/train.npz \
  --validation inputs/seed42/val.npz --model fusion_full --seed 42 \
  --output runs/seed42/fusion_full
python3 -m uefusion predict-fusion --dataset /data/prepared \
  --fit-dir runs/seed42/fusion_full --ran outputs/seed42/ran_test_clean.npz \
  --core outputs/seed42/core_test_clean.npz --output outputs/seed42/fusion_test.npz
```

The seven `--model` choices are in METHODS.md. Fitting requires OOF training
outputs and full-training model-selection outputs; it rejects overlapping fitting
and selection vehicles. Test predictions require same-seed full-fit clean domain
outputs, and do not read test targets. Only the scoring stage joins future labels.

To continue a historical selected GB snapshot, provide `--resume-selected DIR`
containing trusted `model.pkl` and `DONE.json` with the original per-axis histories.
The selected validation MSE is checked before continuation. A new fit without this
option is explicitly a fresh fit, not a disguised recreation of historical reuse.

## Centralized and sensitivity APIs

`train --model central44 --seed 42 --fold full` is the pooled source-feature
reference; `central42` excludes the two speed inputs and preserves the separate
historical comparator's architecture and recipe. Both select by augmented
validation coordinate MSE. Use the ordinary `infer` command for their outputs.

For retention, call `uefusion.cell_state.retention_core` on the full chronological
Core report stream with the original float64-derived floor mask. Substitute those
15 Core fields in a separate cache, fit Core only, and reuse the primary RAN
outputs. All vehicle roles, seeds, GB budgets and selection conditions stay fixed.
No alternative retained-cell RAN model is part of this comparison. Preserve the
original primary cache. The published run reuses the earlier seed-42 Core full fit;
a fresh rerun must label this provenance difference.

For speed sensitivity, `uefusion.speed.generate` accepts original float64 source
speed, report times/keys, a condition (`sigma`, `bias`), a realization seed and
causal source mapping. `source_map` resolves delayed reports; `common_exact_cohort`
defines the common larger-stress cohort. Apply only the received speed/change
columns to fixed trained predictors. These APIs do not schedule study inference.
The published sensitivities did not retrain on noisy speed. Saved-output result
verification is the preferred way to reproduce the paper's reported numbers.

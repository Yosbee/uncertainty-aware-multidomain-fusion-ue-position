#!/usr/bin/env python3
"""Recompute saved-coordinate results; never fit or execute a prediction model."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import re
import zipfile
import numpy as np

SEEDS = tuple(range(42, 47))
MODELS = ('RAN', 'Core', 'fusion_means', 'fusion_rvar', 'fusion_cvar',
          'fusion_full', 'ran_post', 'ran_speed', 'ran_core_raw', 'Central44', 'Central42')
METRICS = ('n', 'vehicles', 'mean_m', 'p95_m', 'mse_m2', 'ran_failures', 'rescues',
           'new_harms', 'final_failures', 'net_reduction', 'net_reduction_pct')
KEYS = ('family', 'condition', 'seed', 'realization', 'model', 'stratum')
STRATA = ('no_floor', 'partial_floor', 'all_floor')


def require(test, message):
    if not test:
        raise ValueError(message)


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


class PrimaryStore:
    """Read a directory or ZIP without extracting members or executing archive code."""
    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(self.path) if self.path.is_file() else None
        names = self.zip.namelist() if self.zip else [str(p.relative_to(self.path)) for p in self.path.rglob('*.npz')]
        self.names = names

    def load(self, name):
        matches = [p for p in self.names if Path(p).name == name]
        require(len(matches) == 1, f'Expected exactly one {name}, found {len(matches)}')
        if self.zip:
            return read_npz(io.BytesIO(self.zip.read(matches[0])))
        return read_npz(self.path / matches[0])

    def close(self):
        if self.zip:
            self.zip.close()


def id_order(source_ids, target_ids):
    """Return indices that align unique source IDs to unique target IDs."""
    source_ids, target_ids = np.asarray(source_ids), np.asarray(target_ids)
    require(source_ids.ndim == target_ids.ndim == 1, 'IDs must be one-dimensional')
    require(len(np.unique(source_ids)) == len(source_ids), 'Duplicate source IDs')
    require(len(np.unique(target_ids)) == len(target_ids), 'Duplicate target IDs')
    order = np.argsort(source_ids)
    slots = np.searchsorted(source_ids[order], target_ids)
    require(np.all(slots < len(order)), 'Missing target ID')
    aligned = order[slots]
    require(np.array_equal(source_ids[aligned], target_ids), 'Missing target ID')
    return aligned


def contract_identity(contract):
    label = contract['label_available']
    require(label.dtype == bool and label.ndim == 1, 'label_available must be a Boolean vector')
    require(len(np.unique(contract['opportunity_ids'])) == len(label), 'Duplicate opportunity IDs')
    for key in ('opportunity_ids', 'scored_window_ids', 'vehicle_ids', 'input_end', 'target_time'):
        require(contract[key].shape == label.shape, f'Contract shape mismatch: {key}')
    truth = contract.get('labelled_truth', contract.get('truth'))
    require(truth is not None and truth.shape == (int(label.sum()), 2), 'Truth shape mismatch')
    require(np.isfinite(truth).all(), 'Non-finite truth')
    require(np.all(contract['scored_window_ids'][~label] == -1), 'Unlabelled samples must have scored ID -1')
    require(np.array_equal(contract['target_time'], contract['input_end'] + 5), 'Target is not exactly t+5')
    ids = contract['scored_window_ids'][label]
    id_order(ids, ids)
    return {**{k: contract[k][label] for k in ('scored_window_ids', 'vehicle_ids', 'input_end', 'target_time')}, 'truth': truth}


def check_identity(candidate, reference):
    order = id_order(reference['scored_window_ids'], candidate['scored_window_ids'])
    for key in ('vehicle_ids', 'input_end', 'target_time', 'truth'):
        require(np.array_equal(candidate[key], reference[key][order]), f'Identity mismatch: {key}')
    return order


def large_cohort(contract):
    """The large-stress paper result uses 220,918 exact-source-time windows."""
    identity = contract_identity(contract)
    mask = contract['exact_lag_common']
    require(mask.dtype == bool and mask.shape == (len(identity['truth']),), 'Invalid exact_lag_common mask')
    require(int(mask.sum()) == 220918, 'Large-speed cohort must contain 220918 exact-lag windows')
    require(len(np.unique(identity['vehicle_ids'][mask])) == 4010, 'Large-speed cohort must contain 4010 vehicles')
    return identity, mask


def load_floor(path, identity):
    floor = read_npz(path)
    order = id_order(floor['scored_window_ids'], identity['scored_window_ids'])
    for key in ('vehicle_ids', 'input_end', 'target_time'):
        require(np.array_equal(floor[key][order], identity[key]), f'Floor identity mismatch: {key}')
    count = floor['floor_count'][order]
    prior = floor['prior_nonfloor_cell'][order]
    require(np.all((count >= 0) & (count <= 10)), 'Floor count must be between 0 and 10')
    masks = {'all': np.ones(len(count), bool), 'no_floor': count == 0,
             'partial_floor': (count > 0) & (count < 10), 'all_floor': count == 10,
             'all_floor_prior_cell': (count == 10) & prior,
             'all_floor_no_prior_cell': (count == 10) & ~prior}
    stored_names = {'partial_floor': 'mixed', **{k: k for k in masks if k != 'partial_floor'}}
    for key, old in stored_names.items():
        require(np.array_equal(masks[key], floor[old][order]), f'Inconsistent floor mask: {old}')
    for key in ('endpoint_floor', 'endpoint_nonfloor'):
        masks[key] = floor[key][order]
    require(np.array_equal(masks['endpoint_floor'], ~masks['endpoint_nonfloor']), 'Endpoint masks are not complements')
    return masks


def errors_from_predictions(pred, truth, selection=None):
    result = {}
    for model in MODELS:
        if model not in pred:
            continue
        coordinates = pred[model] if selection is None else pred[model][selection]
        require(coordinates.shape == truth.shape, f'Prediction shape mismatch: {model}')
        require(np.isfinite(coordinates).all(), f'Non-finite prediction: {model}')
        result[model] = np.linalg.norm(coordinates.astype(np.float64) - truth.astype(np.float64), axis=1)
    require('RAN' in result and 'fusion_full' in result, 'RAN and fusion_full predictions are required')
    return result


def score(errors, masks, identity, family, condition, seed, realization):
    rows, contributions = [], []
    baseline = errors['RAN']
    for stratum, mask in masks.items():
        require(mask.any(), f'Empty stratum: {stratum}')
        ran_fail = baseline[mask] > 50
        for model, vector in errors.items():
            e = vector[mask]
            fail = e > 50
            rescues, harms = int((ran_fail & ~fail).sum()), int((~ran_fail & fail).sum())
            net = rescues - harms
            require(int(fail.sum()) == int(ran_fail.sum()) - net, 'Failure accounting mismatch')
            rows.append(dict(family=family, condition=condition, seed=seed, realization=realization,
                model=model, stratum=stratum, n=int(mask.sum()), vehicles=len(np.unique(identity['vehicle_ids'][mask])),
                mean_m=float(e.mean()), p95_m=float(np.quantile(e, .95)), mse_m2=float(np.mean(e*e)),
                ran_failures=int(ran_fail.sum()), rescues=rescues, new_harms=harms,
                final_failures=int(fail.sum()), net_reduction=net,
                net_reduction_pct=float(100*net/ran_fail.sum()) if ran_fail.any() else 0.0))
        pairs = [('RAN', 'fusion_full'), ('ran_speed', 'fusion_full'), ('fusion_means', 'fusion_full'),
                 ('fusion_means', 'fusion_rvar'), ('fusion_means', 'fusion_cvar'),
                 ('fusion_rvar', 'fusion_full'), ('fusion_full', 'ran_core_raw')]
        for reference, model in pairs:
            if reference not in errors or model not in errors:
                continue
            delta = errors[reference] - errors[model]
            total = float(delta.mean())
            contribution = float(delta[mask].sum()/len(delta))
            contributions.append(dict(family=family, condition=condition, seed=seed, realization=realization,
                reference=reference, model=model, stratum=stratum, n=int(mask.sum()),
                stratum_fraction=float(mask.mean()), within_stratum_reduction_m=float(delta[mask].mean()),
                contribution_m=contribution, total_reduction_m=total,
                share_of_seed_total_pct=float(100*contribution/total) if total else 0.0))
    return rows, contributions


def mean_rows(rows, keys, numeric):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, group in sorted(groups.items()):
        out.append({**dict(zip(keys, key)), **{field: float(np.mean([r[field] for r in group])) for field in numeric}, 'replicates': len(group)})
    return out


def summarize(rows):
    keys = ('family', 'condition', 'seed', 'model', 'stratum')
    per_seed = mean_rows(rows, keys, METRICS)
    summary = mean_rows(per_seed, ('family', 'condition', 'model', 'stratum'), METRICS)
    index = defaultdict(list)
    for row in per_seed:
        index[tuple(row[k] for k in ('family', 'condition', 'model', 'stratum'))].append(row)
    for row in summary:
        part = index[tuple(row[k] for k in ('family', 'condition', 'model', 'stratum'))]
        row['training_seeds'] = len(part)
        row['net_reduction_pct_ratio_mean_counts'] = 100*row['net_reduction']/row['ran_failures'] if row['ran_failures'] else 0.0
        for metric in ('mean_m', 'p95_m', 'mse_m2'):
            row[metric+'_seed_sd'] = float(np.std([r[metric] for r in part], ddof=1)) if len(part)>1 else 0.0
    return per_seed, summary


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def primary_results(store, identity, label, masks):
    rows, contributions = [], []
    for seed in SEEDS:
        pred = store.load(f'test_seed{seed}.npz')
        if seed == 42:
            pred.update(store.load('centralized_seed42.npz'))
        errors = errors_from_predictions(pred, identity['truth'], label)
        r, c = score(errors, masks, identity, 'primary', 'nominal', seed, 0)
        rows.extend(r); contributions.extend(c)
    return rows, contributions


def retention_results(directory, primary_identity, masks, store, primary_label):
    rows, contributions = [], []
    directory = Path(directory)
    for seed in SEEDS:
        path = directory / f'seed{seed}' / 'predictions.npz'
        pred = read_npz(path)
        order = check_identity(pred, primary_identity)
        current_masks = {k: v[order] for k,v in masks.items()}
        original = store.load(f'test_seed{seed}.npz')
        for key in ('RAN', 'ran_variances'):
            require(np.array_equal(pred[key], original[key][primary_label][order]), f'Retention changed fixed RAN: {key}')
        errors = errors_from_predictions(pred, pred['truth'])
        r,c = score(errors, current_masks, pred, 'retention', 'nominal', seed, 0)
        rows.extend(r);contributions.extend(c)
    return rows, contributions


def speed_results(directory, family, primary_identity, masks):
    directory = Path(directory)
    contract = read_npz(directory / 'scoring_contract.npz')
    if family == 'large_speed':
        identity, cohort = large_cohort(contract)
    else:
        identity = contract_identity(contract)
        cohort = np.ones(len(identity['truth']), bool)
        require(len(identity['truth']) == 272201, 'Small-speed cohort must contain all 272201 labelled windows')
    order = check_identity(identity, primary_identity)
    current_masks = {k: v[order][cohort] for k,v in masks.items()}
    selected_identity = {k:v[cohort] for k,v in identity.items()}
    label = contract['label_available']
    rows, contributions = [], []
    conditions = {'nominal':1, 'noise_0.141':3, 'noise_0.205':3} if family=='small_speed' else {
        'nominal':1,'noise_0.5':3,'noise_1':3,'noise_2':3,'bias_minus1':1,'bias_plus1':1,
        'delay_1':1,'delay_3':1,'delay_5':1,'noise_1_delay_3':3}
    for seed in SEEDS:
        files = sorted((directory / f'seed{seed}').glob('*/predictions.npz'))
        grouped = defaultdict(list)
        for path in files:
            match = re.fullmatch(r'(.+)_r(\d+)', path.parent.name)
            require(match is not None, 'Unrecognized speed condition directory')
            grouped[match[1]].append((int(match[2]),path))
        require(set(grouped)==set(conditions), f'Incomplete or extra conditions in {family}, seed {seed}')
        nominal_ran = None
        for condition in ['nominal'] + sorted(set(conditions)-{'nominal'}):
            require(len(grouped[condition])==conditions[condition], f'Wrong realization count for {condition}')
            require(len({x[0] for x in grouped[condition]})==conditions[condition], 'Duplicate realization')
            for realization,path in grouped[condition]:
                pred = read_npz(path)
                aligned = id_order(pred['opportunity_ids'], contract['opportunity_ids'])
                pred = {k:v[aligned][label][cohort] for k,v in pred.items() if k in MODELS}
                if nominal_ran is None:
                    nominal_ran = pred['RAN'].copy()
                require(np.array_equal(pred['RAN'],nominal_ran), 'Speed condition changed fixed RAN predictions')
                errors = errors_from_predictions(pred,selected_identity['truth'])
                r,c = score(errors,current_masks,selected_identity,family,condition,seed,realization)
                rows.extend(r);contributions.extend(c)
    return rows,contributions


def check_expected(rows, path):
    expected = list(csv.DictReader(Path(path).open()))
    index = {tuple(str(r[k]) for k in KEYS): r for r in rows}
    require(len(index)==len(rows),'Duplicate output row')
    available = {r['family'] for r in rows}
    cells = 0
    for row in expected:
        if row['family'] not in available:
            continue
        key = tuple(row[k] for k in KEYS)
        require(key in index, f'Missing expected output row {key}')
        actual = float(index[key][row['metric']])
        require(np.isclose(actual,float(row['value']),rtol=1e-9,atol=1e-7), f'Expected-value mismatch {key} {row["metric"]}: {actual} != {row["value"]}')
        cells += 1
    require(cells>0, 'No expected cells matched the selected family')
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['primary','floor','retention','speed','all'],required=True)
    parser.add_argument('--primary',type=Path,required=True,help='Primary numerical ZIP or extracted directory')
    parser.add_argument('--floor',type=Path,help='window_strata.npz')
    parser.add_argument('--retention',type=Path,help='Directory containing seed42..46/predictions.npz')
    parser.add_argument('--small-speed',type=Path,help='Directory containing scoring_contract.npz and seed42..46')
    parser.add_argument('--large-speed',type=Path,help='Corrected results directory containing scoring_contract.npz and seed42..46')
    parser.add_argument('--expected',type=Path,help='Optional release expected-results CSV')
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    store = PrimaryStore(args.primary)
    try:
        contract = store.load('test_identity.npz')
        identity = contract_identity(contract)
        require(len(identity['truth'])==272201, 'Primary must contain 272201 labelled windows')
        require(len(np.unique(identity['vehicle_ids']))==4032, 'Primary must contain 4032 vehicles')
        label = contract['label_available']
        require(len(label)==286657,'Primary must contain 286657 opportunities')
        masks = {'all':np.ones(len(identity['truth']),bool)}
        if args.mode != 'primary':
            require(args.floor is not None,'--floor is required for stratified modes')
            masks = load_floor(args.floor,identity)
            require([int(masks[k].sum()) for k in STRATA]==[118284,141363,12554],'Unexpected primary floor counts')
        rows, contributions = [], []
        if args.mode in ('primary','floor','all'):
            r,c=primary_results(store,identity,label,masks);rows+=r;contributions+=c
        if args.mode in ('retention','all'):
            require(args.retention is not None,'--retention is required')
            r,c=retention_results(args.retention,identity,masks,store,label);rows+=r;contributions+=c
        if args.mode in ('speed','all'):
            require(args.small_speed is not None or args.large_speed is not None,'A speed results directory is required')
            for directory,family in [(args.small_speed,'small_speed'),(args.large_speed,'large_speed')]:
                if directory:
                    r,c=speed_results(directory,family,identity,masks);rows+=r;contributions+=c
        per_seed,summary=summarize(rows)
        cnum=('n','stratum_fraction','within_stratum_reduction_m','contribution_m','total_reduction_m','share_of_seed_total_pct')
        cseed=mean_rows(contributions,('family','condition','seed','reference','model','stratum'),cnum)
        csummary=mean_rows(cseed,('family','condition','reference','model','stratum'),cnum)
        for row in csummary:
            row['share_of_mean_total_pct']=100*row['contribution_m']/row['total_reduction_m'] if row['total_reduction_m'] else 0.0
        checked = check_expected(rows,args.expected) if args.expected else 0
        args.out.mkdir(parents=True,exist_ok=True)
        for name,table in [('per_realization',rows),('per_seed',per_seed),('summary',summary),
                           ('contributions_per_seed',cseed),('contributions_summary',csummary)]:
            write_csv(args.out/(name+'.csv'),table)
        result={'status':'PASS','expected_numeric_cells':checked,'per_realization_rows':len(rows),
                'aggregation':'Realization metrics averaged within seed, then seeds averaged; no prediction ensembling.',
                'p95':'Linear-interpolated 95th percentile within each seed and realization.',
                'failure_threshold_m':50,'failure_definition':'Euclidean error > 50 m; success <= 50 m.',
                'no_training_inference_or_noise_generation':True,
                'scope_exclusions':['bootstrap confidence intervals','raw radio/scene regeneration','model retraining']}
        (args.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:
        store.close()


if __name__=='__main__':
    main()

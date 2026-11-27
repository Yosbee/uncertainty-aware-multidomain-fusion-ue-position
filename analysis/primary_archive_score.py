#!/usr/bin/env python3
"""Recompute numerical evidence from portable saved coordinates and targets.

No model, checkpoint, external dataset, network, or original workspace is used.
The supplied expected tables are checked only after the scores are recomputed.
"""
from pathlib import Path
import argparse, csv, hashlib, json, platform, sys, time, tempfile, zipfile
import numpy as np

ROOT = None
TEMPORARY = None
SEEDS = list(range(42, 47))
MODELS = ['RAN', 'Core', 'fusion_full', 'fusion_means', 'fusion_rvar', 'fusion_cvar', 'ran_post', 'ran_speed', 'ran_core_raw']
TOLERANCES = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200]
METRICS = ['mean_m', 'median_m', 'p95_m', 'p99_m', 'own_worst_1pct_mean_m', 'mse_m2', 'vehicle_equal_mean_m']

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load(name):
    with np.load(ROOT / 'data' / name, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def read(name):
    with (ROOT / 'expected' / name).open(newline='') as f:
        return list(csv.DictReader(f))

def write(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def metrics(e, vehicle=None):
    n = len(e)
    k = int(np.ceil(.01 * n))
    row = dict(n=n, mean_m=float(e.mean()), median_m=float(np.median(e)),
               p95_m=float(np.quantile(e, .95)), p99_m=float(np.quantile(e, .99)),
               own_worst_1pct_mean_m=float(np.partition(e, n-k)[-k:].mean()),
               mse_m2=float(np.mean(e**2)))
    if vehicle is not None:
        _, inv = np.unique(vehicle, return_inverse=True)
        row['vehicle_equal_mean_m'] = float(np.mean(np.bincount(inv, weights=e) / np.bincount(inv)))
    return row

def compare(actual, expected, keys, fields):
    index = {tuple(str(r[k]) for k in keys): r for r in actual}
    assert len(index) == len(actual), ('Duplicate keys', keys)
    cells = 0
    for old in expected:
        key = tuple(str(old[k]) for k in keys)
        assert key in index, ('Missing comparison row', key)
        new = index[key]
        for field in fields:
            np.testing.assert_allclose(float(new[field]), float(old[field]), rtol=1e-10, atol=1e-8,
                                       err_msg=str((key, field)))
            cells += 1
    return cells

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path, required=True, help='Original numerical archive or extracted directory')
    parser.add_argument('--out', type=Path, default=Path.cwd() / 'recomputed')
    args = parser.parse_args()
    global ROOT, TEMPORARY
    if args.primary.is_file():
        TEMPORARY = tempfile.TemporaryDirectory(prefix='uefusion-primary-')
        root = Path(TEMPORARY.name)
        with zipfile.ZipFile(args.primary) as archive:
            for member in archive.infolist():
                destination = (root / member.filename).resolve()
                if not destination.is_relative_to(root.resolve()):
                    raise ValueError('Unsafe archive member')
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Archive symlinks are not supported')
            archive.extractall(root)
    else:
        root = args.primary
    identities = list(root.rglob('test_identity.npz'))
    if len(identities) != 1:
        raise ValueError('Expected exactly one test_identity.npz')
    ROOT = identities[0].parent.parent
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    checked_files = 0
    for line in (ROOT / 'manifest.sha256').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        assert sha(ROOT / relative) == expected, ('Hash mismatch', relative)
        checked_files += 1

    ident = load('test_identity.npz')
    labelled = ident['label_available']
    ids = ident['scored_window_ids'][labelled]
    truth = ident['labelled_truth'].astype(np.float64)
    vehicle = ident['vehicle_ids'][labelled]
    assert len(labelled) == 286657 and labelled.sum() == 272201
    assert len(np.unique(ident['opportunity_ids'])) == len(labelled)
    assert np.array_equal(np.sort(ids), np.arange(len(ids)))
    assert np.all(ident['scored_window_ids'][~labelled] == -1)
    assert np.array_equal(ident['target_time'], ident['input_end'] + 5)
    assert truth.shape == (len(ids), 2) and np.isfinite(truth).all()
    val_ident = load('validation_identity.npz')
    yval = val_ident['truth'].astype(np.float64)
    assert len(yval) == len(val_ident['window_ids']) == 133806
    assert len(np.unique(val_ident['window_ids'])) == len(yval)

    all_metrics, validation, transitions, diagnostic, contributions = [], [], [], [], []
    errors = {}
    for seed in SEEDS:
        pred = load(f'test_seed{seed}.npz')
        vp = load(f'validation_seed{seed}.npz')
        err = {}
        for model in MODELS:
            assert pred[model].shape == (len(labelled), 2)
            assert np.isfinite(pred[model]).all()
            err[model] = np.linalg.norm(pred[model][labelled].astype(np.float64) - truth, axis=1)
            ev = np.linalg.norm(vp[model].astype(np.float64) - yval, axis=1)
            validation.append(dict(seed=seed, model=model, **metrics(ev)))
        midpoint = (pred['RAN'][labelled].astype(float) + pred['Core'][labelled]) / 2
        err['midpoint'] = np.linalg.norm(midpoint - truth, axis=1)
        err['oracle'] = np.minimum(err['RAN'], err['Core'])
        errors[seed] = err
        groups = {'all': np.ones(len(ids), bool)}
        for base in ['RAN', 'Core']:
            order = np.argsort(-err[base], kind='stable')
            for q in [.01, .1]:
                mask = np.zeros(len(ids), bool)
                mask[order[:int(np.ceil(q * len(ids)))]] = True
                groups[f'{base}_top{int(q*100)}'] = mask
            groups[f'{base}_rest90'] = ~groups[f'{base}_top10']
        for model, e in err.items():
            for group, mask in groups.items():
                all_metrics.append(dict(seed=seed, model=model, group=group, **metrics(e[mask], vehicle[mask])))
            if model in ['midpoint', 'oracle']:
                continue
            for ref in ['RAN', 'Core']:
                for threshold in TOLERANCES:
                    fail, mine = err[ref] > threshold, e > threshold
                    rescued, harmed = int((fail & ~mine).sum()), int((~fail & mine).sum())
                    assert int(mine.sum()) == int(fail.sum()) - rescued + harmed
                    transitions.append(dict(seed=seed, model=model, reference=ref, threshold_m=threshold,
                                            reference_failures=int(fail.sum()), rescued=rescued,
                                            harmed=harmed, final_failures=int(mine.sum())))

        rv = pred['ran_variances'][labelled].astype(float)
        vv = vp['ran_variances'].astype(float)
        assert np.isfinite(rv).all() and np.all(rv > 0)
        assert np.isfinite(vv).all() and np.all(vv > 0)
        u, vu = np.sqrt(rv.sum(axis=1)), np.sqrt(vv.sum(axis=1))
        edges = np.quantile(vu, [.2, .4, .6, .8])
        binid = np.searchsorted(edges, u, side='left')
        keep = u <= np.quantile(vu, .25)
        de = {k: err[k] for k in ['RAN', 'Core', 'fusion_full', 'ran_speed', 'ran_core_raw']}
        de['bypass_q25'] = np.where(keep, err['RAN'], err['fusion_full'])
        dg = [('RAN_above_tolerance', str(t), err['RAN'] > t) for t in TOLERANCES]
        error_edges = [0, 5, 10, 20, 50, 100, 200, float('inf')]
        for lo, hi in zip(error_edges[:-1], error_edges[1:]):
            dg.append(('RAN_error_bin', f'{lo}-{int(hi) if np.isfinite(hi) else "inf"}',
                       (err['RAN'] > lo) & (err['RAN'] <= hi)))
        dg.extend(('uncertainty_bin', str(j+1), binid == j) for j in range(5))
        for family, group, mask in dg:
            assert mask.any()
            rmean = float(err['RAN'][mask].mean())
            for model, e in de.items():
                es = e[mask]
                diff = err['RAN'][mask] - es
                diagnostic.append(dict(seed=seed, family=family, group=group, model=model,
                    n=int(mask.sum()), share_all_pct=100*float(mask.mean()),
                    mean_m=float(es.mean()), median_m=float(np.median(es)), p95_m=float(np.quantile(es, .95)),
                    mean_reduction_vs_RAN_m=float(diff.mean()),
                    relative_mean_reduction_vs_RAN_pct=100*float(diff.mean())/rmean,
                    fraction_individual_error_improved_pct=100*float(np.mean(diff > 0)),
                    fraction_individual_error_worsened_pct=100*float(np.mean(diff < 0))))
            if family == 'uncertainty_bin':
                contributions.append(dict(seed=seed, group=int(group),
                    contribution_m=float(mask.mean()) * float((err['RAN'][mask]-err['fusion_full'][mask]).mean())))
        print(f'Recomputed seed {seed}: {len(ids)} labelled windows', flush=True)

    summary = []
    for model in list(MODELS) + ['midpoint', 'oracle']:
        for group in sorted(groups):
            part = [r for r in all_metrics if r['model'] == model and r['group'] == group]
            row = dict(model=model, group=group)
            for metric in METRICS:
                values = np.array([r[metric] for r in part])
                for suffix, value in [('', values.mean()), ('_seed_sd', values.std(ddof=1)),
                                      ('_seed_min', values.min()), ('_seed_max', values.max())]:
                    row[metric+suffix] = float(value)
            summary.append(row)

    bootstrap = load('vehicle_bootstrap_draws.npz')
    vehicles, inv = np.unique(vehicle, return_inverse=True)
    assert np.array_equal(vehicles, bootstrap['vehicles'])
    weights = bootstrap['multiplicities']
    assert weights.shape == (1000, len(vehicles))
    assert np.all(weights >= 0) and np.all(weights.sum(axis=1) == len(vehicles))
    denominator = weights @ np.bincount(inv)
    paired = []
    for ref in [k for k in MODELS if k != 'fusion_full']:
        delta = np.array([errors[s]['fusion_full'] - errors[s][ref] for s in SEEDS])
        squared = np.array([errors[s]['fusion_full']**2 - errors[s][ref]**2 for s in SEEDS])
        row = dict(model='fusion_full', reference=ref, mean_delta_m=float(delta.mean()),
                   mean_delta_seed_sd=float(delta.mean(axis=1).std(ddof=1)),
                   mean_improving_seeds=int(np.sum(delta.mean(axis=1) < 0)),
                   improved_sample_fraction_mean=float(np.mean(delta < 0)), mse_delta_m2=float(squared.mean()),
                   p95_delta_m=float(np.mean([np.quantile(errors[s]['fusion_full'], .95)-np.quantile(errors[s][ref], .95) for s in SEEDS])))
        for label, values in [('mean', delta), ('mse', squared)]:
            samples = weights @ np.bincount(inv, weights=values.mean(axis=0)) / denominator
            lo, hi = np.quantile(samples, [.025, .975])
            row[label+'_conditional_vehicle_ci_low'] = float(lo)
            row[label+'_conditional_vehicle_ci_high'] = float(hi)
        for seed, value in zip(SEEDS, delta.mean(axis=1)):
            row[f'seed{seed}_mean_delta_m'] = float(value)
        paired.append(row)

    central = []
    for model, position in load('centralized_seed42.npz').items():
        assert position.shape == (len(labelled), 2) and np.isfinite(position).all()
        e = np.linalg.norm(position[labelled].astype(np.float64)-truth, axis=1)
        central.append(dict(seed=42, model=model, split='test', group='ALL', **metrics(e, vehicle)))

    checked_cells = 0
    checked_cells += compare(all_metrics, read('per_seed_metrics.csv'), ['seed', 'model', 'group'], ['n']+METRICS)
    checked_cells += compare(summary, read('summary.csv'), ['model', 'group'],
                             [m+s for m in METRICS for s in ['', '_seed_sd', '_seed_min', '_seed_max']])
    checked_cells += compare(validation, read('per_seed_validation.csv'), ['seed', 'model'], ['n']+METRICS[:-1])
    checked_cells += compare(transitions, read('per_seed_transitions.csv'),
                             ['seed', 'model', 'reference', 'threshold_m'],
                             ['reference_failures', 'rescued', 'harmed', 'final_failures'])
    checked_cells += compare(paired, read('paired.csv'), ['model', 'reference'],
                             [k for k in paired[0] if k not in ['model', 'reference']])
    checked_cells += compare(diagnostic, read('group_metrics_per_fit.csv'), ['seed', 'family', 'group', 'model'],
                             [k for k in diagnostic[0] if k not in ['seed', 'family', 'group', 'model']])
    cexpected = [r for r in read('centralized_performance.csv') if r['split'] == 'test' and r['group'] == 'ALL' and r['model'] in ['Central44', 'Central42']]
    checked_cells += compare(central, cexpected, ['model', 'split', 'group'], ['n', 'mean_m', 'median_m', 'p95_m', 'mse_m2', 'vehicle_equal_mean_m'])
    contrib = {str(j): float(np.mean([r['contribution_m'] for r in contributions if r['group'] == j])) for j in range(1, 6)}
    gain = float(sum(contrib.values()))
    original = json.loads((ROOT / 'expected/MEAN_GAIN_CONTRIBUTION.json').read_text())
    for k, v in contrib.items():
        np.testing.assert_allclose(v, original['group_contributions_m'][k], rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(gain, original['total_mean_reduction_m'], rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(100*contrib['5']/gain, original['highest_group_share_of_total_mean_reduction_pct'], rtol=1e-10, atol=1e-8)
    checked_cells += 7

    for name, rows in [('per_seed_metrics', all_metrics), ('summary', summary), ('validation', validation),
                       ('transitions', transitions), ('paired', paired), ('conditional_groups', diagnostic),
                       ('centralized_single_fit', central), ('mean_gain_contributions', contributions)]:
        write(out / (name+'.csv'), rows)
    by = {(r['model'], r['group']): r for r in summary}
    headings = {m: {k: round(by[m, 'all'][k], 2) for k in ['mean_m', 'p95_m', 'mse_m2']} for m in MODELS}
    report = dict(status='PASS', package_id='primary_numerical_archive',
                  manifest_files_verified=checked_files, expected_numeric_cells_verified=checked_cells,
                  opportunities=286657, labelled_windows_per_fit=len(ids), unlabelled_opportunities=int((~labelled).sum()),
                  test_vehicles_with_labels=len(vehicles), learning_seeds=SEEDS,
                  models_recomputed=MODELS+['midpoint', 'oracle'], centralized_single_fit_models=['Central44', 'Central42'],
                  all_errors_recomputed_from_target_and_predicted_coordinates=True,
                  fixed_tail_definition='Stable descending RAN or Core error; ceil(q*N) records within each fit',
                  own_tail_definition='Each evaluated model selects its own ceil(0.01*N) worst errors',
                  aggregation='Compute each fit metric first, then arithmetic mean; sample SD uses ddof=1; no ensemble',
                  percentile_method='NumPy quantile default linear interpolation',
                  conditional_bootstrap='1000 preserved shared whole-vehicle draws, conditional on the fixed fits',
                  variance_group_definition='Validation-only quintiles of sqrt(RAN variance_x+variance_y); test assignment by searchsorted(side=left)',
                  variance_group_gain_m=contrib, highest_variance_group_gain_share_pct=100*contrib['5']/gain,
                  headline_rounded=headings,
                  RAN_fixed_worst1_mean_m=by['RAN', 'RAN_top1']['mean_m'],
                  fusion_on_RAN_fixed_worst1_mean_m=by['fusion_full', 'RAN_top1']['mean_m'],
                  fusion_own_worst1_mean_m=by['fusion_full', 'all']['own_worst_1pct_mean_m'],
                  new_model_training=0, new_model_inference=0, speed_perturbations_reproduced=False,
                  python=platform.python_version(), numpy=np.__version__, elapsed_seconds=time.monotonic()-start)
    (out / 'CHECK.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    try:
        main()
    finally:
        if TEMPORARY is not None:
            TEMPORARY.cleanup()

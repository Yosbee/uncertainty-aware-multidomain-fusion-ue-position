"""Exact-time windows and role-aware report preprocessing.

The prepared input cache contains engineered observations, not model predictions.
Only fitting vehicles determine imputation means. Standardization is fitted later
on fitting windows, including their repeated reports, matching the study recipe.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .observations import BS, vehicle_keys, normal, uniform, regenerate_radio
from .fusion import CORE_COLS

RAN_COLS = ([f'rss_bs_{b}' for b in BS] + [f'sinr_bs_{b}' for b in BS]
            + ['rss_dbm','sinr_db'] + [f'serving_cell_{i}' for i in range(6)]
            + [f'd_rss_bs_{b}' for b in BS] + [f'd_sinr_bs_{b}' for b in BS]
            + ['rss_range','sinr_range','rss_margin'])
ALL_COLS = list(dict.fromkeys(RAN_COLS + CORE_COLS))
MODEL_COLS = {'ran':RAN_COLS,'core':CORE_COLS,'central44':ALL_COLS,
              'central42':[c for c in ALL_COLS if c not in ('speed','delta_speed')]}


def validate_reports(vehicles, times):
    vehicles=np.asarray(vehicles,dtype=str); times=np.asarray(times)
    if len(vehicles)!=len(times) or not np.isfinite(times).all() or np.any(times!=times.astype(np.int64)):
        raise ValueError('Finite integer report timestamps and aligned vehicle IDs are required')
    order=np.lexsort((times,vehicles))
    if not np.array_equal(order,np.arange(len(times))):
        raise ValueError('Reports must be sorted by vehicle ID then timestamp')
    if np.any((vehicles[1:]==vehicles[:-1]) & (times[1:]<=times[:-1])):
        raise ValueError('Duplicate or non-increasing vehicle/time report keys')


def exact_windows(vehicles,times,target_vehicles,target_times,target_xy):
    """Return all past-eligible windows and mark exact raw t+5 label availability.

    No requirement is imposed on the five intervening future observations. Targets
    must come from the unfiltered coordinate table, not the filtered report table.
    """
    vehicles=np.asarray(vehicles,dtype=str);times=np.asarray(times)
    validate_reports(vehicles,times)
    times=times.astype(np.int64)
    target_times=np.asarray(target_times)
    if not np.isfinite(target_times).all() or np.any(target_times!=target_times.astype(np.int64)):
        raise ValueError('Raw target timestamps must be finite integer seconds')
    if len(times)<10: ends=np.empty(0,np.int64)
    else: ends=np.flatnonzero((vehicles[9:]==vehicles[:-9]) & ((times[9:]-times[:-9])==9))+9
    lookup=ends[:,None]+np.arange(-9,1)
    if len(ends) and (not np.all(times[lookup]==times[ends,None]+np.arange(-9,1)) or not np.all(vehicles[lookup]==vehicles[ends,None])):
        raise ValueError('Nonconsecutive input window')
    raw_index=pd.MultiIndex.from_arrays([np.asarray(target_vehicles,dtype=str),np.asarray(target_times,np.int64)])
    if not raw_index.is_unique: raise ValueError('Duplicate raw coordinate keys')
    target_times=times[ends]+5
    rows=raw_index.get_indexer(pd.MultiIndex.from_arrays([vehicles[ends],target_times]))
    available=rows>=0; y=np.full((len(ends),2),np.nan,np.float32)
    coords=np.asarray(target_xy,np.float32)
    if coords.shape!=(len(raw_index),2) or not np.isfinite(coords).all(): raise ValueError('Invalid raw targets')
    y[available]=coords[rows[available]]
    return dict(report_indices=lookup.astype(np.int32),vehicle_ids=vehicles[ends],input_end=times[ends],target_time=target_times,y=y,label_available=available)


def augment_radio(clean, columns, vehicles, times, seed=2026100900):
    """Primary cached perturbation: no change to supplied speed or cell state.

    Missing primitives remain NaN until the specific fold's fitting-report means
    are known. Perturbations are shared by report key, not generated per window.
    """
    keys=vehicle_keys(vehicles);zero=np.zeros(len(keys),np.int64);ix={c:i for i,c in enumerate(columns)}
    radio_active=uniform(keys,zero,seed,200)>=.3
    radio_sigma=6*uniform(keys,zero,seed,201)*radio_active
    missing_rate=.2*uniform(keys,zero,seed,202)*radio_active
    qvalues=np.array([0.,1.,2.,5.]);q=qvalues[np.minimum((4*uniform(keys,zero,seed,203)).astype(int),3)]*radio_active
    out=np.asarray(clean,np.float32).copy()
    for j,kind in enumerate(('rss','sinr')):
        for b,bs in enumerate(BS):
            stream=j*6+b; col=ix[f'{kind}_bs_{bs}']
            value=clean[:,col].astype(np.float64)+radio_sigma*normal(keys,times,seed,300+stream)
            if kind=='rss':
                active=q>0;value[active]=np.round(value[active]/q[active])*q[active]
            missing=uniform(keys,times,seed,400+stream)<missing_rate
            value[missing]=np.nan;out[:,col]=value
    return out


def vehicle_roles(path, fold):
    roles=json.loads(Path(path).read_text())
    if str(fold)=='full': fit=roles['full_fit_vehicles'];dev=roles['validation_roles']['model_selection'];held=[]
    else:
        f=roles['folds'][int(fold)];fit=f['fit_vehicles'];dev=f['internal_dev_vehicles'];held=f['held_out_vehicles']
    fit,dev,held=[list(map(str,x)) for x in (fit,dev,held)]
    if set(fit)&set(dev) or set(fit+dev)&set(held): raise ValueError('Overlapping fit/development/held-out roles')
    if set(fit+dev+held)&set(map(str,roles['test_vehicles'])): raise ValueError('Test vehicles entered a fitting role')
    return fit,dev,held


class PreparedData:
    def __init__(self, root):
        self.root=Path(root); self.columns=json.loads((self.root/'feature_columns.json').read_text())
        if len(self.columns)!=len(set(self.columns)) or not set(ALL_COLS)<=set(self.columns): raise ValueError('Prepared cache must contain the exact 44 primary features')
        if set(self.columns)&{'x','y','vx','vy'}: raise ValueError('Observed position histories are outside this input contract')
        self.clean=np.load(self.root/'clean_reports.npy',mmap_mode='r',allow_pickle=False)
        self.vehicles=np.load(self.root/'vehicle_ids.npy',mmap_mode='r',allow_pickle=False).astype(str)
        self.times=np.load(self.root/'timestamps.npy',mmap_mode='r',allow_pickle=False)
        validate_reports(self.vehicles,self.times)
        if self.clean.shape!=(len(self.times),len(self.columns)) or not np.isfinite(self.clean).all(): raise ValueError('Invalid clean report cache')
        self.keys=vehicle_keys(self.vehicles)

    def windows(self,split,vehicles=None,scored_only=True,include_targets=True):
        if scored_only and not include_targets: raise ValueError('Scored-only selection requires target availability')
        with np.load(self.root/'windows'/f'{split}.npz',allow_pickle=False) as z:
            keys=z.files if include_targets else ['report_indices','vehicle_ids','input_end','target_time']
            w={k:z[k] for k in keys}
        mask=np.ones(len(w['vehicle_ids']),bool)
        if vehicles is not None: mask &= np.isin(w['vehicle_ids'],vehicles)
        if scored_only: mask &= w['label_available']
        ids=np.flatnonzero(mask); w={k:v[ids] for k,v in w.items()};w['window_ids']=ids
        ix=w['report_indices']
        if ix.shape!=(len(ids),10) or np.any(ix<0) or np.any(ix>=len(self.times)): raise ValueError('Invalid report lookup')
        if not np.array_equal(self.times[ix],w['input_end'][:,None]+np.arange(-9,1)): raise ValueError('Window does not contain exact t-9,...,t reports')
        if not np.all(self.vehicles[ix]==w['vehicle_ids'][:,None]) or not np.all(w['target_time']==w['input_end']+5): raise ValueError('Window identity or horizon mismatch')
        if scored_only and not np.isfinite(w['y']).all(): raise ValueError('Training/scoring requires exact raw future labels')
        return w

    def resolve(self,fit_vehicles,augment=False,means=None):
        mask=np.isin(self.vehicles,fit_vehicles)
        if means is None and not mask.any(): raise ValueError('No fitting reports')
        mean=self.clean[mask].astype(np.float64).mean(0) if means is None else np.asarray(means)
        if mean.shape!=(len(self.columns),): raise ValueError('Imputation means shape mismatch')
        if not augment: return np.array(self.clean,copy=True),mean
        x=np.array(np.load(self.root/'augmented_unimputed_reports.npy',mmap_mode='r',allow_pickle=False),copy=True)
        ci=[self.columns.index(c) for c in CORE_COLS]
        if not np.array_equal(x[:,ci],self.clean[:,ci]): raise ValueError('Primary radio augmentation must preserve Core inputs')
        for q in ('rss','sinr'):
            for b in BS:
                j=self.columns.index(f'{q}_bs_{b}');bad=~np.isfinite(x[:,j]);x[bad,j]=mean[j]
        x=regenerate_radio(x,self.columns,self.keys)
        if not np.isfinite(x).all(): raise ValueError('Unresolved nonfinite observations')
        return x,mean

    def materialize(self,reports,windows,columns):
        take=[self.columns.index(c) for c in columns]
        out=np.empty((len(windows['report_indices']),10,len(take)),np.float32)
        for i in range(0,len(out),20000): out[i:i+20000]=reports[windows['report_indices'][i:i+20000]][:,:,take]
        return out


def prepare(reports_csv,targets_csv,split_json,output):
    """Build a portable cache from retained engineered reports and raw labels."""
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    reports=pd.read_csv(reports_csv,dtype={'vehicle_id':str}).sort_values(['vehicle_id','timestamp'])
    targets=pd.read_csv(targets_csv,dtype={'vehicle_id':str})
    split=json.loads(Path(split_json).read_text())
    split=split.get('source_splits',split)
    if not all(k in split for k in ('train','val','test')): raise ValueError('Split file needs train/val/test or source_splits')
    all_windows=exact_windows(reports.vehicle_id.to_numpy(),reports.timestamp.to_numpy(),targets.vehicle_id.to_numpy(),targets.timestamp.to_numpy(),targets[['scene_x','scene_y']].to_numpy())
    output.mkdir(parents=True);(output/'windows').mkdir()
    clean=reports[ALL_COLS].to_numpy(np.float32);vehicles=reports.vehicle_id.to_numpy(str);times=reports.timestamp.to_numpy(np.int64)
    np.save(output/'clean_reports.npy',clean);np.save(output/'vehicle_ids.npy',vehicles);np.save(output/'timestamps.npy',times)
    np.save(output/'augmented_unimputed_reports.npy',augment_radio(clean,ALL_COLS,vehicles,times))
    (output/'feature_columns.json').write_text(json.dumps(ALL_COLS,indent=2)+'\n')
    seen=set()
    for name in ('train','val','test'):
        vids=set(map(str,split[name]))
        if seen&vids: raise ValueError('Split vehicle overlap')
        seen|=vids;mask=np.isin(all_windows['vehicle_ids'],list(vids))
        np.savez(output/'windows'/f'{name}.npz',**{k:v[mask] for k,v in all_windows.items()})

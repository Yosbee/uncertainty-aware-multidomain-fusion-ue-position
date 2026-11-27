"""Explicit research commands. Merely importing the package performs no work."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
from .data import PreparedData,CORE_COLS
from .fusion import MODELS,features


def load_npz(path):
    with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in z.files}


def assemble(args):
    from .inference import assemble_oof
    data=PreparedData(args.dataset);w=data.windows('train')
    arrays=[load_npz(p) for p in args.outputs]
    metadata=[json.loads(Path(p).with_suffix('.json').read_text()) for p in args.outputs]
    conditions={m['condition'] for m in metadata};models={m['model'] for m in metadata};seeds={m['seed'] for m in metadata}
    if len(conditions)!=1 or len(models)!=1 or len(seeds)!=1 or {m['fold'] for m in metadata}!=set(map(str,range(5))): raise ValueError('Fold models/seeds/conditions do not match')
    p,v=assemble_oof(w['window_ids'],w['vehicle_ids'],w['input_end'],w['target_time'],arrays,metadata)
    out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez(out,predictions=p,variances=v,**{k:w[k] for k in ('window_ids','vehicle_ids','input_end','target_time')})
    out.with_suffix('.json').write_text(json.dumps({'kind':'vehicle_oof','model':next(iter(models)),'seed':next(iter(seeds)),'condition':next(iter(conditions))},indent=2)+'\n')


def fusion_inputs(args):
    data=PreparedData(args.dataset);w=data.windows(args.split)
    outputs=[load_npz(p) for p in (args.ran,args.core)]
    metadata=[json.loads(Path(p).with_suffix('.json').read_text()) for p in (args.ran,args.core)]
    if [m['model'] for m in metadata]!=['ran','core'] or metadata[0]['seed']!=metadata[1]['seed']: raise ValueError('Wrong domain or mismatched training seed')
    if any(m['condition']!='augmented' for m in metadata): raise ValueError('GB fitting uses augmented-domain outputs')
    if args.split=='train':
        if any(m.get('kind')!='vehicle_oof' for m in metadata): raise ValueError('GB training requires vehicle OOF outputs')
        kind='vehicle_oof'
    else:
        if any(m.get('fold')!='full' for m in metadata): raise ValueError('GB selection requires full-training outputs')
        kind='full_training_model_selection'
        selected=set(metadata[0]['development_vehicles'])
        if selected!=set(metadata[1]['development_vehicles']): raise ValueError('Mismatched model-selection vehicles')
        w=data.windows('val',list(selected))
    pieces=[]
    for a in outputs:
        index={int(x):i for i,x in enumerate(a['window_ids'])}
        if len(index)!=len(a['window_ids']): raise ValueError('Duplicate prediction identities')
        rows=np.array([index[int(i)] for i in w['window_ids']])
        for key in ('vehicle_ids','input_end','target_time'):
            if not np.array_equal(a[key][rows],w[key]): raise ValueError('Prediction/target join mismatch')
        pieces.extend([a['predictions'][rows],a['variances'][rows]])
    core=data.materialize(data.clean,w,CORE_COLS).reshape(len(w['y']),150)
    out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez(out,summary=np.column_stack(pieces).astype(np.float32),core=core,**{k:w[k] for k in ('y','window_ids','vehicle_ids','input_end','target_time')})
    out.with_suffix('.json').write_text(json.dumps({'kind':kind,'seed':metadata[0]['seed'],'radio_condition':'augmented','columns':'RAN position, RAN variance, Core position, Core variance'},indent=2)+'\n')


def fit_fusion(args):
    from .boosting import fit_axis
    train=load_npz(args.train);val=load_npz(args.validation)
    meta=[json.loads(Path(p).with_suffix('.json').read_text()) for p in (args.train,args.validation)]
    if [m['kind'] for m in meta]!=['vehicle_oof','full_training_model_selection']: raise ValueError('Wrong fitting/selection output roles')
    if any(m['seed']!=args.seed for m in meta): raise ValueError('Seed mismatch')
    if set(train['vehicle_ids'])&set(val['vehicle_ids']): raise ValueError('Fitting/selection vehicle overlap')
    F=features(args.model,train['summary'],train['core']);Fv=features(args.model,val['summary'],val['core'])
    y=train['y'];yv=val['y']
    if y.shape!=(len(F),2) or yv.shape!=(len(Fv),2) or not all(np.isfinite(v).all() for v in (F,Fv,y,yv)): raise ValueError('Invalid fusion arrays')
    out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    oldmodels=[None,None];oldinfos=[None,None]
    if args.resume_selected:
        prior=Path(args.resume_selected)
        with (prior/'model.pkl').open('rb') as f:oldmodels=pickle.load(f)
        oldinfos=json.loads((prior/'DONE.json').read_text())['fit_info']
        for a in range(2):
            actual=np.mean((oldmodels[a].predict(Fv)-yv[:,a])**2)
            if abs(actual-oldinfos[a]['best_val_mse'])>=1e-6: raise ValueError('Prior selected snapshot or validation data differs')
    models=[];infos=[]
    for axis in range(2):
        model,info=fit_axis(F,y[:,axis],Fv,yv[:,axis],args.seed,out,axis,oldmodels[axis],oldinfos[axis]);models.append(model);infos.append(info)
    with (out/'model.pkl').open('wb') as f:pickle.dump(models,f)
    (out/'DONE.json').write_text(json.dumps({'seed':args.seed,'model':args.model,'fit_info':infos,'n_features':F.shape[1],'test_access':False,'fit_vehicles':sorted(set(train['vehicle_ids'])), 'development_vehicles':sorted(set(val['vehicle_ids'])), 'restart_source':str(args.resume_selected) if args.resume_selected else None},indent=2)+'\n')


def predict_fusion(args):
    """Apply the selected coordinate models without loading future coordinates."""
    root=Path(args.fit_dir);done=json.loads((root/'DONE.json').read_text())
    data=PreparedData(args.dataset);w=data.windows('test',scored_only=False,include_targets=False)
    sources=[load_npz(p) for p in (args.ran,args.core)]
    metadata=[json.loads(Path(p).with_suffix('.json').read_text()) for p in (args.ran,args.core)]
    if [m['model'] for m in metadata]!=['ran','core']: raise ValueError('Wrong domain output order')
    if any(m['condition']!='clean' or m['fold']!='full' or m['seed']!=done['seed'] for m in metadata): raise ValueError('Primary test requires same-seed full-model clean outputs')
    for m in metadata:
        if set(w['vehicle_ids'])&set(m['fit_vehicles']+m['development_vehicles']): raise ValueError('Test vehicle used for domain fit/selection')
    if set(w['vehicle_ids'])&set(done['fit_vehicles']+done['development_vehicles']): raise ValueError('Test vehicle used for GB fit/selection')
    pieces=[]
    for a in sources:
        index={int(x):i for i,x in enumerate(a['window_ids'])}
        if len(index)!=len(a['window_ids']): raise ValueError('Duplicate output IDs')
        rows=np.array([index[int(i)] for i in w['window_ids']])
        for key in ('vehicle_ids','input_end','target_time'):
            if not np.array_equal(a[key][rows],w[key]): raise ValueError('Prediction identity/timestamp mismatch')
        pieces.extend([a['predictions'][rows],a['variances'][rows]])
    summary=np.column_stack(pieces).astype(np.float32)
    core=data.materialize(data.clean,w,CORE_COLS).reshape(len(summary),150)
    F=features(done['model'],summary,core)
    if F.shape[1]!=done['n_features'] or not np.isfinite(F).all(): raise ValueError('Feature interface differs from fitted fusion')
    with (root/'model.pkl').open('rb') as f:models=pickle.load(f)
    if len(models)!=2: raise ValueError('Two coordinate regressors required')
    predictions=np.column_stack([model.predict(F) for model in models])
    if not np.isfinite(predictions).all(): raise ValueError('Nonfinite fused predictions')
    out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez(out,predictions=predictions,**{k:w[k] for k in ('window_ids','vehicle_ids','input_end','target_time')})
    out.with_suffix('.json').write_text(json.dumps({'model':done['model'],'seed':done['seed'],'radio_condition':'clean','target_coordinates_loaded':False},indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare',help='Build a cache from retained engineered reports and unfiltered exact-time targets')
    p.add_argument('--reports',required=True);p.add_argument('--targets',required=True);p.add_argument('--splits',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('train',help='Fit one domain/fold or seed-42 centralized reference')
    p.add_argument('--dataset',required=True);p.add_argument('--splits',required=True);p.add_argument('--model',choices=['ran','core','central44','central42'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--fold',choices=['full','0','1','2','3','4'],required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
    p=sub.add_parser('infer',help='Infer explicit held-out/full-fit outputs from a trained checkpoint')
    p.add_argument('--dataset',required=True);p.add_argument('--checkpoint-dir',required=True);p.add_argument('--split',choices=['train','val','test'],required=True);p.add_argument('--condition',choices=['clean','augmented'],required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    p=sub.add_parser('assemble-oof',help='Join exactly one held-out domain output per scored training window')
    p.add_argument('--dataset',required=True);p.add_argument('--outputs',nargs=5,required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('fusion-inputs',help='Align domain outputs with target and raw Core history')
    p.add_argument('--dataset',required=True);p.add_argument('--ran',required=True);p.add_argument('--core',required=True);p.add_argument('--split',choices=['train','val'],required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('fit-fusion',help='Fit the two coordinate GB models using fixed validation selection')
    p.add_argument('--train',required=True);p.add_argument('--validation',required=True);p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--seed',type=int,choices=range(42,47),required=True);p.add_argument('--output',required=True);p.add_argument('--resume-selected',help='Optional prior selected GB bundle with validation history; never silently reused')
    p=sub.add_parser('predict-fusion',help='Apply a fitted fusion to clean test-domain outputs without reading targets')
    p.add_argument('--dataset',required=True);p.add_argument('--fit-dir',required=True);p.add_argument('--ran',required=True);p.add_argument('--core',required=True);p.add_argument('--output',required=True)
    args=ap.parse_args()
    if args.command=='prepare':
        from .data import prepare
        prepare(args.reports,args.targets,args.splits,args.output)
    elif args.command=='train':
        from .train import run
        run(args.dataset,args.splits,args.model,args.seed,args.fold,args.output,args.device,args.threads)
    elif args.command=='infer':
        from .inference import run
        run(args.dataset,args.checkpoint_dir,args.split,args.condition,args.output,args.device)
    elif args.command=='assemble-oof':assemble(args)
    elif args.command=='fusion-inputs':fusion_inputs(args)
    elif args.command=='fit-fusion':fit_fusion(args)
    elif args.command=='predict-fusion':predict_fusion(args)

if __name__=='__main__':main()

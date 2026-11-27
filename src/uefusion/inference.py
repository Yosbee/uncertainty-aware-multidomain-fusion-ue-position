"""Explicit checkpoint inference and exactly-once vehicle-held-out assembly."""
from pathlib import Path
import hashlib,json,pickle
import numpy as np
import torch
from .data import PreparedData
from .models import make_model


@torch.no_grad()
def predict(model,X,device):
    model.eval();positions=[];variances=[]
    for i in range(0,len(X),1024):
        value=model(torch.from_numpy(X[i:i+1024]).to(device),return_uncertainty=True)
        p,v=value;positions.append(p.cpu().numpy());variances.append(v.cpu().numpy())
    p=np.concatenate(positions);v=np.concatenate(variances)
    if not np.isfinite(p).all() or not np.isfinite(v).all() or np.any(v<=0): raise ValueError('Invalid domain outputs')
    return p,v


def run(dataset,checkpoint_dir,split,condition,output,device='cpu'):
    root=Path(checkpoint_dir);cfg=json.loads((root/'config.json').read_text());done=json.loads((root/'DONE.json').read_text())
    checkpoint=root/done['selected_checkpoint']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=done['checkpoint_sha256']: raise ValueError('Checkpoint hash mismatch')
    model_name=cfg['model'];is_domain=model_name in ('ran','core')
    if cfg['fold']!='full' and split!='train': raise ValueError('Fold inference is limited to its held-out training vehicles')
    if split=='test' and condition!='clean': raise ValueError('Primary test inference uses clean reports; sensitivities are separate')
    data=PreparedData(dataset)
    if cfg['fold']!='full': vehicles=cfg['held_out_vehicles']
    elif split=='val': vehicles=cfg['development_vehicles']
    elif split=='train': vehicles=cfg['fit_vehicles']
    else: vehicles=None
    windows=data.windows(split,vehicles,scored_only=False,include_targets=False)
    if len(windows['vehicle_ids'])==0: raise ValueError('No selected windows')
    if split=='test' and set(windows['vehicle_ids'])&set(cfg['fit_vehicles']+cfg['development_vehicles']): raise ValueError('Test vehicle leakage')
    with (root/'preprocessing.pkl').open('rb') as f: bundle=pickle.load(f)
    reports,_=data.resolve(bundle['fit_vehicles'],augment=condition=='augmented',means=bundle['imputation_means'])
    cols=bundle['feature_columns'];X=data.materialize(reports,windows,cols)
    X=bundle['features'].transform(X.reshape(-1,len(cols))).reshape(X.shape).astype(np.float32)
    net=make_model(model_name);net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict']);net.to(device)
    if is_domain: p,v=predict(net,X,device)
    else:
        with torch.no_grad():
            net.eval();p=np.concatenate([net(torch.from_numpy(X[i:i+1024]).to(device)).cpu().numpy() for i in range(0,len(X),1024)])
        v=None
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    arrays={k:windows[k] for k in ('window_ids','vehicle_ids','input_end','target_time')};arrays['predictions']=p
    if v is not None:arrays['variances']=v
    np.savez(output,**arrays)
    output.with_suffix('.json').write_text(json.dumps({'model':model_name,'condition':condition,'fold':cfg['fold'],'seed':cfg['seed'],'fit_vehicles':cfg['fit_vehicles'],'development_vehicles':cfg['development_vehicles'],'held_out_vehicles':cfg['held_out_vehicles'],'checkpoint_sha256':done['checkpoint_sha256']},indent=2)+'\n')


def assemble_oof(window_ids,vehicle_ids,input_end,target_time,fold_outputs,fold_metadata):
    """Gather exactly one prediction per requested training window, never by row position alone."""
    window_ids=np.asarray(window_ids);vehicle_ids=np.asarray(vehicle_ids,dtype=str)
    if len(np.unique(window_ids))!=len(window_ids): raise ValueError('Duplicate destination IDs')
    index={int(v):i for i,v in enumerate(window_ids)};coverage=np.zeros(len(window_ids),np.int8)
    p=np.empty((len(window_ids),2),np.float32);v=np.empty_like(p)
    if len(fold_outputs)!=5 or len(fold_metadata)!=5: raise ValueError('Five fold outputs required')
    for arrays,meta in zip(fold_outputs,fold_metadata):
        ids=np.asarray(arrays['window_ids']);selected=np.isin(ids,window_ids);ids=ids[selected]
        if len(np.unique(ids))!=len(ids): raise ValueError('Duplicate fold IDs')
        rows=np.array([index[int(j)] for j in ids]);actual=set(vehicle_ids[rows])
        if actual & set(meta['fit_vehicles']+meta['development_vehicles']): raise ValueError('OOF heldout vehicles entered fitting/checkpoint selection')
        if not actual<=set(meta['held_out_vehicles']): raise ValueError('Wrong heldout vehicle fold')
        if not np.array_equal(np.asarray(arrays['vehicle_ids'])[selected],vehicle_ids[rows]): raise ValueError('OOF identity mismatch')
        for key,expected in [('input_end',input_end),('target_time',target_time)]:
            if not np.array_equal(np.asarray(arrays[key])[selected],np.asarray(expected)[rows]): raise ValueError('OOF timestamp mismatch')
        coverage[rows]+=1;p[rows]=arrays['predictions'][selected];v[rows]=arrays['variances'][selected]
    if not np.all(coverage==1): raise ValueError('Missing or multiply-covered OOF windows')
    if not np.isfinite(p).all() or not np.isfinite(v).all() or np.any(v<=0): raise ValueError('Invalid OOF outputs')
    return p,v

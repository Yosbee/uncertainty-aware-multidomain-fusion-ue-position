"""Explicit fitting entrypoint; never loads test windows."""
from pathlib import Path
import hashlib,json,pickle
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from .data import PreparedData,vehicle_roles,MODEL_COLS
from .models import make_model
from .training_loop import fit_phase


def run(dataset,splits,model,seed,fold,output,device='cpu',threads=4):
    if seed not in range(42,47): raise ValueError('Manuscript training seeds are 42--46')
    domain=model in ('ran','core')
    if not domain and (str(fold)!='full' or seed!=42): raise ValueError('Centralized references are full-fit seed-42 comparisons')
    fit,dev,held=vehicle_roles(splits,fold)
    data=PreparedData(dataset);fw=data.windows('train',fit);dw=data.windows('val' if str(fold)=='full' else 'train',dev)
    if len(fw['y'])<2 or len(dw['y'])<1: raise ValueError('Insufficient fit/development windows')
    out=Path(output)
    if out.exists(): raise FileExistsError('Existing fit is never silently overwritten')
    out.mkdir(parents=True)
    torch.set_num_threads(threads);torch.manual_seed(seed);np.random.seed(seed);device=torch.device(device)
    cols=MODEL_COLS[model]
    reports,means=data.resolve(fit,augment=not domain)
    X=data.materialize(reports,fw,cols);Xv=data.materialize(reports,dw,cols);del reports
    sc=StandardScaler().fit(X.reshape(-1,len(cols)))
    X=sc.transform(X.reshape(-1,len(cols))).reshape(X.shape).astype(np.float32)
    Xv=sc.transform(Xv.reshape(-1,len(cols))).reshape(Xv.shape).astype(np.float32)
    with (out/'preprocessing.pkl').open('wb') as f: pickle.dump({'features':sc,'feature_columns':cols,'imputation_means':means,'fit_vehicles':fit,'report_columns':data.columns},f)
    metadata={'model':model,'seed':seed,'fold':str(fold),'fit_vehicles':fit,'development_vehicles':dev,'held_out_vehicles':held,'fit_windows':len(X),'development_windows':len(Xv),'test_access':False,'domain_training':domain,'splits_sha256':hashlib.sha256(Path(splits).read_bytes()).hexdigest()}
    (out/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    net=make_model(model).to(device)
    checkpoint=fit_phase(net,X,fw['y'],Xv,dw['y'],device,domain,seed,out,'clean' if domain else 'central_augmented',40)
    if domain:
        del X,Xv
        reports,again=data.resolve(fit,augment=True)
        np.testing.assert_array_equal(means,again)
        X=data.materialize(reports,fw,cols);Xv=data.materialize(reports,dw,cols);del reports
        X=sc.transform(X.reshape(-1,len(cols))).reshape(X.shape).astype(np.float32)
        Xv=sc.transform(Xv.reshape(-1,len(cols))).reshape(Xv.shape).astype(np.float32)
        checkpoint=fit_phase(net,X,fw['y'],Xv,dw['y'],device,True,seed,out,'augmented',8 if model=='ran' else 5)
    (out/'DONE.json').write_text(json.dumps({'selected_checkpoint':checkpoint.name,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'test_access':False},indent=2)+'\n')
    return out

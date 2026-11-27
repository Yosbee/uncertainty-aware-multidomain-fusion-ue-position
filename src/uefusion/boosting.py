"""Per-coordinate histogram GB with validation-selected snapshots."""
from pathlib import Path
import copy, json, os, time, datetime
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def fit_axis(F,y,Fv,yv,seed,out,axis,oldmodel=None,oldinfo=None,rule=None):
 rule = rule or dict(chunk=100, patience=5, cap=12000);chunk=rule['chunk'];patience=rule['patience'];cap=rule['cap']
 started=time.monotonic();replayed=0
 if oldinfo and oldinfo['iters_fitted']-oldinfo['best_iter']>=patience*chunk:
  return oldmodel,{**oldinfo,'action':'REUSED_CONVERGED','new_iterations':0,'seconds_this_stage':0}
 if oldmodel is not None:
  m=copy.deepcopy(oldmodel);best=oldinfo['best_val_mse'];bestn=oldinfo['best_iter'];n=bestn;since=0;final=copy.deepcopy(m)
  curve=oldinfo['val_mse_per_chunk'][:n//chunk];start_n=n
  assert m.n_iter_==n and m.warm_start and m.max_leaf_nodes==63 and m.learning_rate==.1
 else:
  m=HistGradientBoostingRegressor(max_iter=chunk,learning_rate=.1,max_leaf_nodes=63,early_stopping=False,random_state=seed,warm_start=True)
  best=float('inf');bestn=0;n=0;since=0;curve=[];start_n=0
 while n<cap:
  n+=chunk;m.set_params(max_iter=n);m.fit(F,y)
  mse=float(np.mean((m.predict(Fv)-yv)**2));curve.append(mse)
  if oldinfo and n<=oldinfo['iters_fitted']:
   assert abs(mse-oldinfo['val_mse_per_chunk'][n//chunk-1])<1e-6,(n,mse,oldinfo['val_mse_per_chunk'][n//chunk-1])
   replayed+=chunk
  if mse<best:best=mse;bestn=n;since=0;final=copy.deepcopy(m)
  else:since+=1
  save(out/'PROGRESS.json',dict(status='FITTING',seed=seed,axis=axis,iteration=n,best_iteration=bestn,best_val_mse=best,seconds=time.monotonic()-started,pid=os.getpid(),updated_utc=now()))
  print('CHUNK',axis,n,mse,'best',bestn,flush=True)
  if since>=patience:break
 return final,dict(best_iter=bestn,iters_fitted=n,best_val_mse=best,val_mse_per_chunk=curve,action='EXTENDED_CAPPED' if oldinfo else 'NEW_FIT',new_iterations=n-start_n,replayed_old_iterations=replayed,reached_cap=n==cap,patience_satisfied=since>=patience,seconds_this_stage=time.monotonic()-started)

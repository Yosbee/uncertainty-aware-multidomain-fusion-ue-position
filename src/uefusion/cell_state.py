"""Recorded Core feature derivations and retrospective retention rule."""
import numpy as np
# Fixed centroid values actually used to construct the published model inputs.
BSXY = np.array([[260.2,274.1],[-306.9,-60.3],[-21.5,319.6],[-429.3,-336.4],[-137.8,-264.8],[301.4,-10.5]],np.float64)

def derive(cell,times,vehicles,speed,delta_speed):
    """Preserve legacy event/direction conventions; unknown->known is attachment, not handover."""
    n=len(cell);idx=np.arange(n);first=np.r_[True,vehicles[1:]!=vehicles[:-1]]
    starts=np.maximum.accumulate(np.where(first,idx,0));valid=cell>=0
    out=np.zeros((n,15),np.float32)
    out[idx[valid],cell[valid]]=1
    centroid=np.zeros((n,2),np.float64);centroid[valid]=BSXY[cell[valid]];out[:,6:8]=centroid
    out[:,8]=speed;out[:,14]=delta_speed
    prev=np.r_[-1,cell[:-1]]
    ho=(~first)&valid&(prev>=0)&(cell!=prev);out[:,9]=ho
    last=np.maximum.accumulate(np.where(ho,idx,-1));known=last>=starts
    out[known,10]=np.minimum(times[known]-times[last[known]],300)
    direction=np.zeros((n,2));direction[ho]=centroid[ho]-BSXY[prev[ho]]
    for a in range(2):
        last=np.maximum.accumulate(np.where(direction[:,a]!=0,idx,-1));known=last>=starts
        out[known,11+a]=direction[last[known],a]
    changed=first|(cell!=prev);last=np.maximum.accumulate(np.where(changed,idx,0))
    out[valid,13]=np.minimum(times[valid]-times[last[valid]],600)
    out[~valid,:8]=0;out[~valid,9:14]=0
    return out

def hold_cells(cell,floor_tie,vehicles):
    idx=np.arange(len(cell));first=np.r_[True,vehicles[1:]!=vehicles[:-1]]
    starts=np.maximum.accumulate(np.where(first,idx,0))
    last=np.maximum.accumulate(np.where(~floor_tie,idx,-1))
    valid=last>=starts
    result=np.full(len(cell),-1,np.int8);result[valid]=cell[last[valid]]
    assert np.array_equal(result[~floor_tie],cell[~floor_tie])
    return result


def retention_core(original_core, floor_report, times, vehicles):
    """Change only Core's serving-state proxy; RAN outputs must be reused.

    The floor mask must come from original float64 received-power reports. Do not
    reconstruct it from a float32 cache whose near-floor distinctions collapsed.
    """
    original_core=np.asarray(original_core)
    if original_core.shape!=(len(times),15): raise ValueError('Core report shape mismatch')
    cell=np.argmax(original_core[:,:6],axis=1).astype(np.int8)
    if not np.all(original_core[:,:6].sum(1)==1): raise ValueError('Primary cell proxy must be one-hot')
    baseline=derive(cell,np.asarray(times),np.asarray(vehicles),original_core[:,8],original_core[:,14])
    if not np.array_equal(baseline,original_core): raise ValueError('Core derivation does not reproduce source cache')
    held=hold_cells(cell,np.asarray(floor_report,bool),np.asarray(vehicles))
    return derive(held,np.asarray(times),np.asarray(vehicles),original_core[:,8],original_core[:,14])

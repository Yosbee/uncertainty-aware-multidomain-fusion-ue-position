"""Fixed-model speed-error stream, using float64 source differences before float32 conversion."""
import numpy as np
from .observations import normal

def generate(speed64,times,keys,c,rep,mapping):
 before=np.asarray(speed64,np.float64)+c['bias']
 if c['sigma']:before+=c['sigma']*normal(keys,times,rep,410)
 source=np.maximum(0.,before)
 valid=mapping>=0;received=np.full(len(source),np.nan,np.float64);received[valid]=source[mapping[valid]]
 delta=np.full(len(source),np.nan,np.float64);same=np.r_[False,keys[1:]==keys[:-1]]
 delta[(~same)&valid]=0.
 prev=np.maximum(np.arange(len(source))-1,0);ok=same&valid&valid[prev]
 delta[ok]=np.clip(received[ok]-received[prev[ok]],-10,10)
 return np.column_stack([received,delta]).astype(np.float32),source,before


def source_map(times, keys, delay_seconds):
    """Latest same-vehicle source report no later than receipt time minus delay."""
    times=np.asarray(times)
    keys=np.asarray(keys)
    if delay_seconds<0: raise ValueError('Delay must be nonnegative')
    result=np.full(len(times),-1,np.int64)
    starts=np.flatnonzero(np.r_[True,keys[1:]!=keys[:-1]])
    ends=np.r_[starts[1:],len(times)]
    for first,last in zip(starts,ends):
        t=times[first:last]
        if np.any(np.diff(t)<=0): raise ValueError('Source stream must be strictly ordered within vehicle')
        j=np.searchsorted(t,t-delay_seconds,side='right')-1
        valid=j>=0;result[first:last][valid]=j[valid]+first
    return result


def common_exact_cohort(lookup,times,keys,delays=(0,1,3,5)):
    """Availability-only large-stress cohort before any model errors are read.

    Both the current and preceding retained report need exact delayed source
    timestamps. The latter is needed for the supplied speed difference. The first
    report of a vehicle has a zero difference.
    """
    times=np.asarray(times);keys=np.asarray(keys);lookup=np.asarray(lookup)
    valid=np.ones(len(times),bool);same=np.r_[False,keys[1:]==keys[:-1]]
    for delay in delays:
        mapping=source_map(times,keys,delay);ok=mapping>=0
        exact=np.zeros(len(times),bool);exact[ok]=times[mapping[ok]]==times[ok]-delay
        previous=np.r_[False,exact[:-1]]
        valid &= exact & (~same | previous)
    return np.flatnonzero(valid[lookup].all(1))

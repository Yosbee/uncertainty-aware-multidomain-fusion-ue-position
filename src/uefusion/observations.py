"""Stateless report-keyed radio augmentation and derived measurements."""
import hashlib
import numpy as np
BS = [56, 75, 114, 118, 147, 244]

def vehicle_keys(vehicle_ids):
    unique, inverse = np.unique(np.asarray(vehicle_ids,dtype=str),return_inverse=True)
    hashes = np.array([int.from_bytes(hashlib.blake2b(s.encode(),digest_size=8).digest(),'little') for s in unique],dtype=np.uint64)
    return hashes[inverse]

def mix64(values):
    with np.errstate(over='ignore'):
        z = np.asarray(values,dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))

def uniform(keys, times, seed, stream=0):
    # Streams represent primitive fields; fixed keys are independent of array order/chunking.
    with np.errstate(over='ignore'):
        code = mix64(np.asarray(keys,dtype=np.uint64)) ^ mix64(np.asarray(times,dtype=np.uint64))
        code ^= mix64(np.uint64(seed)) ^ mix64(np.uint64(stream + 1))
    bits = mix64(code) >> np.uint64(11)
    return (bits.astype(np.float64) + .5) * (1. / 9007199254740992.)

def normal(keys, times, seed, stream=0):
    return np.sqrt(-2*np.log(uniform(keys,times,seed,2*stream))) * np.cos(2*np.pi*uniform(keys,times,seed,2*stream+1))

def regenerate_radio(features,columns,keys):
    """Derived reporting values; historical cell-state proxy is kept separate."""
    x=np.array(features,copy=True);ix={c:i for i,c in enumerate(columns)}
    rss=x[:,[ix[f'rss_bs_{b}'] for b in BS]];sinr=x[:,[ix[f'sinr_bs_{b}'] for b in BS]]
    best=np.argmax(rss,axis=1);row=np.arange(len(x));sort=np.sort(rss,axis=1)
    state_columns=[f'serving_cell_{j}' for j in range(6)]
    if all(c in ix for c in state_columns):
        state=x[:,[ix[c] for c in state_columns]]
        previous=np.argmax(state,axis=1)
        tied=(state.sum(1)>0)&(rss[row,previous]==rss.max(1))
        # Source float64 distinctions can collapse after float32 conversion or
        # quantization. Use contemporaneously available cell-state proxy to
        # break a reporting tie; no future state or target is accessed.
        best[tied]=previous[tied]
    x[:,ix['rss_dbm']]=rss[row,best];x[:,ix['sinr_db']]=sinr[row,best]
    x[:,ix['rss_range']]=rss.max(1)-rss.min(1);x[:,ix['sinr_range']]=sinr.max(1)-sinr.min(1)
    x[:,ix['rss_margin']]=sort[:,-1]-sort[:,-2]
    same=keys[1:]==keys[:-1]
    for kind in ('rss','sinr'):
        for b in BS:
            values=x[:,ix[f'{kind}_bs_{b}']];delta=np.zeros(len(x))
            delta[1:][same]=np.clip(np.diff(values)[same],-30,30)
            x[:,ix[f'd_{kind}_bs_{b}']]=delta
    return x

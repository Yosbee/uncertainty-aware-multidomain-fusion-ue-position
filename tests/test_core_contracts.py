import json
from pathlib import Path
import numpy as np
import pytest
from uefusion.data import exact_windows,ALL_COLS,RAN_COLS,MODEL_COLS,augment_radio,PreparedData,vehicle_roles
from uefusion.observations import vehicle_keys,normal,regenerate_radio
from uefusion.cell_state import derive,hold_cells,retention_core
from uefusion.fusion import features,MODELS
from uefusion.inference import assemble_oof
from uefusion.speed import generate,source_map,common_exact_cohort


def test_exact_horizon_survives_filtered_future_gap():
    t=np.r_[np.arange(10),15];v=np.repeat('a',len(t))
    w=exact_windows(v,t,['a'],[14],[[2,3]])
    assert w['input_end'].tolist()==[9]
    assert w['target_time'].tolist()==[14]
    assert w['y'].tolist()==[[2.,3.]]
    assert w['report_indices'].tolist()==[list(range(10))]


def test_missing_label_retains_opportunity():
    w=exact_windows(np.repeat('a',11),np.arange(11),['a'],[14],[[2,3]])
    assert w['input_end'].tolist()==[9,10]
    assert w['label_available'].tolist()==[True,False]
    assert np.isnan(w['y'][1]).all()


@pytest.mark.parametrize('times',[np.arange(10)+.1,np.r_[0,0,np.arange(2,10)],np.r_[1,0,np.arange(2,10)]])
def test_invalid_report_times_rejected(times):
    with pytest.raises(ValueError):exact_windows(np.repeat('a',10),times,['a'],[14],[[2,3]])


def test_noninteger_target_rejected():
    with pytest.raises(ValueError):exact_windows(np.repeat('a',10),np.arange(10),['a'],[14.1],[[2,3]])


def test_exact_input_gaps_are_not_windows():
    w=exact_windows(np.repeat('a',10),np.r_[np.arange(9),10],['a'],[15],[[2,3]])
    assert len(w['input_end'])==0


def test_model_boundaries_and_gb_interfaces():
    assert len(RAN_COLS)==35 and len(ALL_COLS)==44 and len(MODEL_COLS['core'])==15
    assert not set(ALL_COLS)&{'x','y','vx','vy'}
    summary=np.arange(24).reshape(3,8);core=np.arange(450).reshape(3,150)
    assert [features(m,summary,core).shape[1] for m in MODELS]==[8,4,6,6,4,24,154]
    np.testing.assert_array_equal(features('ran_speed',summary,core)[:,4:],core.reshape(3,10,15)[:,:,[8,14]].reshape(3,20))


def test_keyed_noise_is_subset_invariant_and_augmentation_preserves_core():
    v=np.repeat(['a','b'],20);t=np.tile(np.arange(20),2);keys=vehicle_keys(v)
    np.testing.assert_array_equal(normal(keys,t,5)[[2,25]],normal(keys[[2,25]],t[[2,25]],5))
    clean=np.zeros((40,44),np.float32)
    for k in ('speed','delta_speed','serving_cell_0'):clean[:,ALL_COLS.index(k)]=1
    augmented=augment_radio(clean,ALL_COLS,v,t)
    corecols=[ALL_COLS.index(c) for c in MODEL_COLS['core']]
    np.testing.assert_array_equal(augmented[:,corecols],clean[:,corecols])


def test_radio_recompute_keeps_cell_proxy_and_tie_break():
    x=np.zeros((3,44),np.float32);x[:,ALL_COLS.index('serving_cell_2')]=1
    x[:,ALL_COLS.index('sinr_bs_114')]=7
    y=regenerate_radio(x,ALL_COLS,np.repeat('a',3))
    assert (y[:,ALL_COLS.index('sinr_db')]==7).all()
    assert (y[:,ALL_COLS.index('serving_cell_2')]==1).all()


def test_retention_unknown_and_prior_state():
    v=np.array(['a']*5+['b']*3);c=np.array([0,2,0,0,3,0,0,4]);floor=np.array([1,0,1,1,0,1,1,0],bool);t=np.array([0,1,2,5,6,0,1,2])
    held=hold_cells(c,floor,v);assert held.tolist()==[-1,2,2,2,3,-1,-1,4]
    base=derive(c,t,v,np.ones(8),np.zeros(8));core=retention_core(base,floor,t,v)
    assert core[:,9].tolist()==[0,0,0,0,1,0,0,0]
    assert core[:,13].tolist()==[0,0,1,4,0,0,0,0]
    assert not core[[0,5,6],:8].any()


def test_speed_difference_uses_original_precision_and_vehicle_boundaries():
    t=np.array([0,1,2,0,1,2]);k=vehicle_keys(['a']*3+['b']*3)
    speed=np.array([1.00000002,1.00000006,3.,2.,2.5,3.])
    stream,_,_=generate(speed,t,k,dict(sigma=0.,bias=0.),0,np.arange(6))
    assert stream[1,1]==np.float32(speed[1]-speed[0])
    assert stream[3,1]==0
    assert source_map(t,k,1).tolist()==[-1,0,1,-1,3,4]


def test_fold_role_overlap_rejected(tmp_path):
    f=tmp_path/'roles.json';f.write_text(json.dumps({'full_fit_vehicles':['a'],'test_vehicles':['t'],'validation_roles':{'model_selection':['a']}}))
    with pytest.raises(ValueError):vehicle_roles(f,'full')


def test_oof_duplicate_and_leakage_rejected():
    ids=np.arange(5);vehicles=np.array(list('abcde'));outputs=[];meta=[]
    for i,v in enumerate(vehicles):
        outputs.append(dict(window_ids=np.array([i]),vehicle_ids=np.array([v]),predictions=np.zeros((1,2)),variances=np.ones((1,2)),input_end=np.array([9]),target_time=np.array([14])))
        meta.append(dict(fit_vehicles=[],development_vehicles=[],held_out_vehicles=[v]))
    p,v=assemble_oof(ids,vehicles,np.repeat(9,5),np.repeat(14,5),outputs,meta);assert p.shape==(5,2)
    meta[0]['development_vehicles']=['a']
    with pytest.raises(ValueError):assemble_oof(ids,vehicles,np.repeat(9,5),np.repeat(14,5),outputs,meta)
    meta[0]['development_vehicles']=[];outputs[1]=outputs[0]
    with pytest.raises(ValueError):assemble_oof(ids,vehicles,np.repeat(9,5),np.repeat(14,5),outputs,meta)


def test_imputation_uses_unique_fitting_reports(tmp_path):
    n=6;clean=np.zeros((n,44),np.float32);clean[:,0]=[1,3,100,200,300,400]
    for c in MODEL_COLS['core']:clean[:,ALL_COLS.index(c)]=0
    augmented=clean.copy();augmented[4,0]=np.nan
    for name,array in [('clean_reports',clean),('augmented_unimputed_reports',augmented),('vehicle_ids',np.array(['a','a','b','b','b','b'])),('timestamps',np.array([0,1,0,1,2,3]))]:np.save(tmp_path/f'{name}.npy',array)
    (tmp_path/'feature_columns.json').write_text(json.dumps(ALL_COLS))
    data=PreparedData(tmp_path);x,means=data.resolve(['a'],True)
    assert means[0]==2 and x[4,0]==2


def test_oof_time_mismatch_rejected():
    ids=np.arange(5);vehicles=np.array(list('abcde'));outputs=[];meta=[]
    for i,v in enumerate(vehicles):
        outputs.append(dict(window_ids=np.array([i]),vehicle_ids=np.array([v]),predictions=np.zeros((1,2)),variances=np.ones((1,2)),input_end=np.array([9]),target_time=np.array([14])))
        meta.append(dict(fit_vehicles=[],development_vehicles=[],held_out_vehicles=[v]))
    outputs[0]['target_time'][0]=15
    with pytest.raises(ValueError,match='timestamp'):assemble_oof(ids,vehicles,np.repeat(9,5),np.repeat(14,5),outputs,meta)


def test_preparation_uses_released_role_schema(tmp_path):
    import pandas as pd
    from uefusion.data import prepare
    reports=pd.DataFrame(np.zeros((10,44)),columns=ALL_COLS);reports['vehicle_id']='a';reports['timestamp']=np.arange(10)
    reports.to_csv(tmp_path/'reports.csv',index=False)
    pd.DataFrame(dict(vehicle_id=['a'],timestamp=[14],scene_x=[2.],scene_y=[3.])).to_csv(tmp_path/'targets.csv',index=False)
    (tmp_path/'roles.json').write_text(json.dumps({'source_splits':{'train':['a'],'val':[],'test':[]}}))
    prepare(tmp_path/'reports.csv',tmp_path/'targets.csv',tmp_path/'roles.json',tmp_path/'cache')
    data=PreparedData(tmp_path/'cache');w=data.windows('train')
    assert w['y'].tolist()==[[2.,3.]]
    (tmp_path/'cache/windows/test.npz').unlink()
    # Fitting data access does not require a test cache.
    assert len(data.windows('train'))>0


def test_predict_fusion_does_not_require_target_arrays(tmp_path):
    import argparse,pickle
    from sklearn.dummy import DummyRegressor
    from uefusion.cli import predict_fusion
    cache=tmp_path/'cache';cache.mkdir();(cache/'windows').mkdir()
    (cache/'feature_columns.json').write_text(json.dumps(ALL_COLS))
    np.save(cache/'clean_reports.npy',np.zeros((10,44),np.float32));np.save(cache/'vehicle_ids.npy',np.repeat('test',10));np.save(cache/'timestamps.npy',np.arange(10))
    # There is deliberately no y or label_available array in this inference input.
    np.savez(cache/'windows/test.npz',report_indices=np.arange(10)[None,:],vehicle_ids=np.array(['test']),input_end=np.array([9]),target_time=np.array([14]))
    fit=tmp_path/'fit';fit.mkdir()
    models=[DummyRegressor(strategy='constant',constant=v).fit(np.zeros((3,8)),np.zeros(3)) for v in (2.,3.)]
    with (fit/'model.pkl').open('wb') as f:pickle.dump(models,f)
    (fit/'DONE.json').write_text(json.dumps({'seed':42,'model':'fusion_full','n_features':8,'fit_vehicles':['train'],'development_vehicles':['val']}))
    for name in ('ran','core'):
        out=tmp_path/f'{name}.npz';np.savez(out,window_ids=np.array([0]),vehicle_ids=np.array(['test']),input_end=np.array([9]),target_time=np.array([14]),predictions=np.zeros((1,2)),variances=np.ones((1,2)))
        out.with_suffix('.json').write_text(json.dumps({'model':name,'seed':42,'fold':'full','condition':'clean','fit_vehicles':['train'],'development_vehicles':['val']}))
    target=tmp_path/'predicted.npz';predict_fusion(argparse.Namespace(dataset=cache,fit_dir=fit,ran=tmp_path/'ran.npz',core=tmp_path/'core.npz',output=target))
    with np.load(target) as result:assert result['predictions'].tolist()==[[2.,3.]]


def test_saved_imputation_means_allow_inference_only_cache(tmp_path):
    clean=np.zeros((10,44),np.float32)
    for name,array in [('clean_reports',clean),('vehicle_ids',np.repeat('test',10)),('timestamps',np.arange(10))]:np.save(tmp_path/f'{name}.npy',array)
    (tmp_path/'feature_columns.json').write_text(json.dumps(ALL_COLS))
    data=PreparedData(tmp_path);x,mean=data.resolve(['unavailable_train_vehicle'],means=np.ones(44))
    assert np.array_equal(x,clean) and np.array_equal(mean,np.ones(44))
    with pytest.raises(ValueError,match='No fitting reports'):data.resolve(['unavailable_train_vehicle'])

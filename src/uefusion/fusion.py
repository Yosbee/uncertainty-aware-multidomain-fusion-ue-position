"""The seven manuscript GB input interfaces."""
import numpy as np
MODELS=['fusion_full','fusion_means','fusion_rvar','fusion_cvar','ran_post','ran_speed','ran_core_raw']
COLS={'fusion_full':list(range(8)),'fusion_means':[0,1,4,5],'fusion_rvar':[0,1,2,3,4,5],'fusion_cvar':[0,1,4,5,6,7],'ran_post':[0,1,2,3]}
CORE_COLS=['serving_cell_0','serving_cell_1','serving_cell_2','serving_cell_3','serving_cell_4','serving_cell_5','cell_centroid_x','cell_centroid_y','speed','handover_flag','time_since_handover','handover_dir_x','handover_dir_y','cell_dwell_time','delta_speed']

def features(model,s,core):
 if model in COLS:return np.asarray(s[:,COLS[model]],dtype=np.float32,order='C')
 if model=='ran_speed':
  j=[CORE_COLS.index(x) for x in ['speed','delta_speed']]
  speed=np.asarray(core).reshape(-1,10,15)[:,:,j].reshape(len(s),20)
  return np.column_stack([s[:,:4],speed]).astype(np.float32)
 assert model=='ran_core_raw'
 return np.column_stack([s[:,:4],core]).astype(np.float32)

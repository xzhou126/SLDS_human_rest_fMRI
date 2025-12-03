import os
os.environ["OMP_NUM_THREADS"] = "1"

import pickle
import numpy as np

K = 6
with open('/home/xzhou126/ssm/Final_model/K6_D10_500subjs_compact_model.pkl', 'rb') as f:
    [model, q, elbos, q_z] = pickle.load(f)

num_roi = model.N
num_subject = len(np.unique(model.tags))
pid = np.unique(model.tags)
    
with open('/home/xzhou126/ssm/data/roi_timeseries_rsfMRI_HCP_held_out', 'rb') as f:
    datas = pickle.load(f)
with open('/home/xzhou126/ssm/data/tags_rsfMRI_HCP_held_out', 'rb') as f:
    tags = pickle.load(f)

datas = [datas[i] for i in range(len(datas)) if tags[i] in pid]  
tags = list(np.concatenate([np.repeat(pid[l], 4) for l in range(len(pid))]))
num_run = len(datas)

C = model.parent.emissions.Cs[0]

r_square_by_state = np.zeros(K)
residual_by_state = dict()
ys_by_state = dict()
for k in range(K): 
    residual_by_state[k] = []
    ys_by_state[k] = []

for i in range(num_run):
    ys = datas[i]
    tag = tags[i]
    zs = q_z[i]

    def one_step_predict(y_t, z_t):
        y_hat = C.dot(model.children[tag].dynamics.As[z_t]).dot(C.T).dot(y_t) + C.dot(model.children[tag].dynamics.bs[z_t]) 
        return y_hat
        
    for t in range(1, len(zs)):
        residual_by_state[zs[t-1]].append(ys[t,:] - one_step_predict(ys[t-1,:], zs[t-1]))
        ys_by_state[zs[t-1]].append(ys[t,:])

for k in range(K):
    residual_by_state[k] = np.array(residual_by_state[k])
    ys_by_state[k] = np.array(ys_by_state[k])
    # assert residual_by_state[k].shape[0] == ys_by_state[k].shape[0]
    r_square_by_state[k] = 1-np.linalg.norm(np.cov(residual_by_state[k].T))/np.linalg.norm(np.cov(ys_by_state[k].T))

with open('r_square_by_state.pkl','wb') as f:
   pickle.dump(r_square_by_state, f)

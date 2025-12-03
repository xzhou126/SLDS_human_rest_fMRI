import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
K = int(sys.argv[1])
D = int(sys.argv[2])
fold = int(sys.argv[3])

import pickle
import numpy as np
from hierarchical_SLDS_V3 import HierarchicalSLDS 
from lds import SLDS
from util import find_permutation
import autograd.numpy.random as npr
import warnings
warnings.filterwarnings("ignore")

with open('roi_timeseries_rsfMRI_HCP_model_selection', 'rb') as f:
    datas = pickle.load(f)

with open('tags_rsfMRI_HCP_model_selection', 'rb') as f:
    tags = pickle.load(f)

subject_id = np.unique(tags)

# 10 folds, 10 subjects each fold
folds = []
for i in range(10):
    folds.append(np.arange(10*i, 10*(i+1), dtype=int))

num_roi = datas[0].shape[1]
emissions_dim = num_roi 

tags_train = list(np.concatenate([np.repeat(subject_id[l], 4) for l in range(100) if l not in folds[fold]]))
datas_train = [datas[i] for i in range(400) if tags[i] in tags_train]

model = HierarchicalSLDS(N=emissions_dim, K=K, D=D, 
                          transitions = "recurrent", emissions="gaussian_orthog", 
                          tags = tags_train, 
                          lmbda=0.1)

elbos, q = model.fit(datas = datas_train, tags = tags_train, num_iters=50, 
                        initialize = True, discrete_state_init_method = 'kmeans')

q_x = q.mean_continuous_states
q_z = []
for x, data, tag in zip(q_x, datas_train, tags_train):
    q_z.append(model.children[tag].most_likely_states(x, data))

del q 
q = []

with open('model_selection_models/K%s_D%s_model-%s.pkl' %(K, D, fold+1),'wb') as f:
    pickle.dump([model, q, elbos, q_z], f)
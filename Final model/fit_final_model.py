#import os
#os.environ["OMP_NUM_THREADS"] = "1"

#import sys
K = 6
D = 10

import pickle
import numpy as np
from hierarchical_SLDS_V3 import HierarchicalSLDS 
from lds import SLDS
from util import find_permutation
import autograd.numpy.random as npr
import warnings
warnings.filterwarnings("ignore")

with open('roi_timeseries_rsfMRI_HCP_held_out', 'rb') as f:
    datas = pickle.load(f)

with open('tags_rsfMRI_HCP_held_out', 'rb') as f:
    tags = pickle.load(f)

subject_id = np.unique(tags)

num_roi = datas[0].shape[1]
emissions_dim = num_roi 
    
num_subjs = 500
subject_id_selected = subject_id[0:num_subjs]    
datas_selected = [datas[i] for i in range(len(datas)) if tags[i] in subject_id_selected]    # collect data arrays for selected subjects
# 4 repeated tags for each subject
tags_selected = list(np.concatenate([np.repeat(subject_id_selected[l], 4) for l in range(len(subject_id_selected))]))

model = HierarchicalSLDS(N=emissions_dim, K=K, D=D, 
                          transitions = "recurrent", emissions="gaussian_orthog", 
                          tags = tags_selected, 
                          lmbda=0.1)

elbos, q = model.fit(datas = datas_selected, tags = tags_selected, num_iters=50, 
                        initialize = True, discrete_state_init_method = 'kmeans')

q_x = q.mean_continuous_states
q_z = []
for x, data, tag in zip(q_x, datas_selected, tags_selected):
    q_z.append(model.children[tag].most_likely_states(x, data))

# save model
with open('Final_model/K%s_D%s_%ssubjs_model.pkl' %(K, D, num_subjs),'wb') as f:
    pickle.dump([model, q, elbos, q_z], f)

# also store a compact version without q
del q 
q = []
with open('Final_model/K%s_D%s_%ssubjs_compact_model.pkl' %(K, D, num_subjs),'wb') as f:
    pickle.dump([model, q, elbos, q_z], f)

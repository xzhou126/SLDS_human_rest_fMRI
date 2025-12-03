import os
os.environ["OMP_NUM_THREADS"] = "1"

import pickle
import numpy as np
import resting_state_summaries as rss

K = 6
D = 10
with open('Final_model/K6_D10_500subjs_compact_model.pkl', 'rb') as f:
    [model, q, elbos, q_z] = pickle.load(f)

num_roi = model.N
num_subject = len(np.unique(model.tags))
pid = np.unique(model.tags)

with open('data/roi_timeseries_rsfMRI_HCP_held_out', 'rb') as f:
    datas = pickle.load(f)
with open('data/tags_rsfMRI_HCP_held_out', 'rb') as f:
    tags = pickle.load(f)

with open('summary_data/q_x.pkl','rb') as f:
    q_x = pickle.load(f)

C = model.emissions.Cs[0]
emission_noise = np.exp(model.parent.emissions.inv_etas[0])
Sigma_e_inv = np.diag(1/emission_noise)
C_lesion = []
for m in range(num_roi):
    C_copy = C.copy()
    C_lesion.append(np.delete(C_copy, m, axis=0))

def roi_to_latent(y, emission_mat = C, ols=True):
    if ols == True:
        x_ols = (emission_mat.T @ (y.T)).T
        return x_ols
    else: # take into account heterogeneity in emission noise 
        emission_mat_wls = np.linalg.inv(emission_mat.T @ Sigma_e_inv @ emission_mat) @ emission_mat.T @ Sigma_e_inv
        x_wls = (emission_mat_wls @ (y.T)).T
        return x_wls

def softmax(hs):
    sum_exp = np.sum(np.exp(hs), axis = 1)
    sum_exp = np.tile(sum_exp[:, np.newaxis, :], (1, K, 1))
    return np.exp(hs)/sum_exp

with open('summary_data/transition_courses_importance.pkl','rb') as f:
    transition_courses = pickle.load(f)
# collect significant transitions
transitions_of_interest = []
K = 6
for k1 in range(K):
    for k2 in range(K):
        if len(transition_courses[k1][k2])>0:
            transitions_of_interest.append([k1, k2])
transitions_of_interest = np.array(transitions_of_interest)
num_transitions = transitions_of_interest.shape[0]

with open('summary_data/state_order.pkl','rb') as f:
   state_order = pickle.load(f)

t_list = np.arange(-5,7)
transition_importance_time_subject= np.zeros((num_subject, num_transitions, len(t_list) , num_roi)) + np.nan
for i in range(num_transitions):
    k1 = state_order[transitions_of_interest[i,0]]
    k2 = state_order[transitions_of_interest[i,1]]

    for s in range(num_subject):
        y_subject = [datas[j] for j in range(len(datas)) if tags[j] == pid[s]]
        z_subject = [q_z[j] for j in range(len(datas)) if tags[j] == pid[s]]
        x_subject = [q_x[j] for j in range(len(datas)) if tags[j] == pid[s]]
            
        log_Q = model.children[pid[s]].transitions.log_Ps 
        Rs = model.children[pid[s]].transitions.Rs 
        def compute_h(xs):
            T = xs.shape[0]
            h = np.zeros((K,K,T))
            for i in range(K):
                for j in range(K):
                    h[i,j,:] = log_Q[i,j] + Rs[j,:].dot(xs.T)
            return h
        
        for t0 in t_list:
            x_t = []
            y_t = []
            pi = []
            pi_lesion = dict()
            y_lesion = dict()

            for m in range(num_roi):
                pi_lesion[m] = []
                y_lesion[m] = []

            for j in range(len(y_subject)):
                ys = y_subject[j]
                zs = z_subject[j]
                xs = x_subject[j]
                T = len(zs)
                
                for t in range(T-1): 
                    if (zs[t] == k1) & (zs[t+1] == k2): # identify transition point
                        if t0<0: # backtrace
                            if (np.array_equal(zs[(t+t0):t], np.repeat(k1,-t0))) & (t+t0>=0):
                                x_t.append(xs[t+t0])
                                y_t.append(ys[t+t0])
                            else:
                                break
                        if t0 == 0:
                            x_t.append(xs[t])
                            y_t.append(ys[t])
                        if t0>0: # forwardtrace
                            if (t+t0<T) & (np.array_equal(zs[(t+1):(t+t0)], np.repeat(k2,t0-1))):
                                x_t.append(xs[t+t0])
                                y_t.append(ys[t+t0])
                            else:
                                break

            if len(x_t)==0:                   
                continue # 
            else: 
                x_t = np.stack(x_t)
                y_t = np.stack(y_t)
                y_lesion = []
                for m in range(num_roi):
                    y = y_t.copy()
                    y_lesion.append(np.delete(y, m, axis=1))
            
                x_lesion = [roi_to_latent(y_lesion[m], emission_mat = C_lesion[m], ols = True) for m in range(num_roi)]
                
                h_t = compute_h(x_t)
                h_odds_t =  h_t[k1,k2,:] - h_t[k1,k1,:]
                h_odds = np.mean(h_odds_t)

                h_lesion_t = [compute_h(x_lesion[m]) for m in range(num_roi)]
                h_lesion_odds_t = [h_lesion_t[m][k1,k2,:] - h_lesion_t[m][k1,k1,:] for m in range(num_roi)]
                h_lesion_odds = np.mean(h_lesion_odds_t, axis = 1)
                
                # transition_importance_time_subject[s,i,t0-t_list[0],:] = h_odds - h_lesion_odds 

                transition_importance_time_subject[s,i,t0-t_list[0],:] = np.mean(h_lesion_odds) - h_lesion_odds

with open('summary_data/transition_importance_time_subject.pkl','wb') as f:
    pickle.dump(transition_importance_time_subject, f)
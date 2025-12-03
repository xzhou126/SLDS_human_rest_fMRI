import numpy as np

def collect_z_bundle(qz):      
    z_bundle = []
    num_run = len(qz)
    for i in range(num_run):
        zs = qz[i]
        T = len(zs)
        current_state = -1 # reset at run beginning
        for t in range(T):
            if t == 0: 
                current_state = zs[0]
                state_duration = 1
            elif zs[t] == current_state: 
                state_duration += 1 
            else: 
                z_bundle.append(np.repeat(current_state, state_duration)) 
                current_state = zs[t]
                state_duration = 1
            if t == len(zs)-1:
                z_bundle.append(np.repeat(current_state, state_duration )) 
                state_duration  = 0 # reset at end of run
    return z_bundle

def collect_y_bundle(ys, z_bundle, K):
    y_bundles = [] # length K, collect bundles for each state 
    for k in range(K):
        y_bundle = [] # bundle of BOLD responses for given state
        for i in range(len(z_bundle)):
            if z_bundle[i][0] == k:
                time_knots = np.cumsum([len(z_bundle[j]) for j in range(i)])
                if len(time_knots)==0:
                    current_time = 0
                else:
                    current_time = time_knots[-1]
                y_bundle.append(ys[current_time:(current_time+len(z_bundle[i])),:])
        y_bundles.append(y_bundle)
    return y_bundles

def activity_evolution(y_bundles, T = 20):
    activity_evolution = dict()
    K = len(y_bundles)
    num_roi = y_bundles[0][0].shape[1]
    for k in range(K):  
        activity_evolution[k] = dict()
        for j in range(num_roi):
            activity_evolution[k][j] = np.zeros(T) 
            for t in range(T):
                # collect timeseries values from bundles that durates at least (t+1) time steps
                bold_t = []
                for i in range(len(y_bundles[k])):
                    if len(y_bundles[k][i])>=(t+1):
                        bold_t.append(y_bundles[k][i][:,j][t])  # activity of jth ROI at time t of state entry
                # calculate mean responses
                activity_evolution[k][j][t] = np.mean(bold_t)
    return(activity_evolution)

# compute state duration from z_bundle
def compute_state_duration(z_bundle, K, method = 'mean'):
    empirical_durations = dict()
    for k in range(K):
        empirical_durations[k] = []
    for i in range(len(z_bundle)):
        current_state = np.unique(z_bundle[i])[0]
        empirical_durations[current_state].append(len(z_bundle[i]))
    state_duration = np.zeros(K)
    for k in range(K): 
        if method == 'mean':
            state_duration[k] = np.mean(empirical_durations[k])
        if method == 'median':
            state_duration[k] = np.median(empirical_durations[k])
    return state_duration

# auc
def compute_auc(activity_evolution, T = 5):
    num_roi = len(activity_evolution)
    auc = np.zeros(num_roi)
    for j in range(num_roi):
        auc[j] = np.mean(activity_evolution[j][1:(T+1)])-activity_evolution[j][0]
    return auc

# covariance evolution
def get_cov_series(y_bundles, K, use_cov = True):
    covariance_series = dict()
    for k in range(K):
        covariance_series[k] = dict()
        for t in range(20):
            # collect all points t times steps into state
            bold_t = []
            for i in range(len(y_bundles[k])):
                if len(y_bundles[k][i])>=(t+1):
                    bold_t.append(y_bundles[k][i][t,:])  # BOLD response vector at t time-step in state
            # need at least length 2 to compute covariance, if not stop adding new covariance values to dictionary
            if len(bold_t)>1:
                if use_cov ==True:
                    covariance_series[k][t] = np.cov(np.column_stack(bold_t))
                else: # use correlation
                    covariance_series[k][t] = np.corrcoef(np.column_stack(bold_t))
            else:
                break
    return covariance_series

def cumulative_cov_series(y_bundles, K):
    covariance_series = dict()
    for k in range(K):
        covariance_series[k] = dict()
        for t in range(20):
            # collect all points within t times steps into state
            bold_t = []
            for i in range(len(y_bundles[k])):
                if len(y_bundles[k][i])>=(t+1):
                    for tt in range(t+1):
                        bold_t.append(y_bundles[k][i][tt,:])  # BOLD response vectors within t time-step in state
            # need at least length 2 to compute covariance, if not stop adding new covariance values to dictionary
            if len(bold_t)>1:
                covariance_series[k][t] = np.cov(np.column_stack(bold_t))
            else:
                break
    return covariance_series

def permute_matrix(mat, pi): # permute matrix according to vector pi
    n = mat.shape[0]
    permuted_mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            permuted_mat[i,j] = mat[pi[i], pi[j]]
    return permuted_mat

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

# compute state probability vector from q_z
def compute_state_probability(q_z, K):
    state_p = np.zeros(K)
    for k in range(K):
        for i in range(len(q_z)):
            state_p[k] += sum(q_z[i] == k)
    T = 0
    for i in range(len(q_z)):
        T += len(q_z[i])    
    state_p = state_p/T
    return state_p

# compute empirical transition matrix from state sequences q_z
def compute_transition_mat(q_z, K):
    transition_mat = np.zeros((K, K)) 
    current_state = -1
    for i in range(len(q_z)):
        for j in range(len(q_z[i])):
            if j == 0:
                current_state = q_z[i][j]
            else: 
                transition_mat[current_state, q_z[i][j]] += 1 
                current_state = q_z[i][j]
    row_sums = np.sum(transition_mat, axis = 1)
    for k in range(K):
        transition_mat[k, ] = transition_mat[k, ]/row_sums[k]            
    return transition_mat

# exit transition matrix (normalize off-diagonals of transition matrix by each row)
def exit_transition(P):
    # P: transition matrix
    K = P.shape[0]
    exit_transition_mat = np.zeros((K, K))
    for k in range(K):
        mask = np.ones(K, dtype = bool)
        mask[k] = 0
        exit_transition_mat[k, mask] = P[k, mask]/sum(P[k, mask])
    return exit_transition_mat
                   
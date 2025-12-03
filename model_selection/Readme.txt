We use 100 HCP subjects for model selection, 10-fold cross validation models (model_selection.py) are fitted in parallel (parallel.sh).
We first select number of states, K, via normalized mutual information (NMI) and adjusted Rand index (ARI) 
We then select number of latent dimensions, D, via explained variance (R^2)
import copy

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.stats import norm
from autograd.misc.optimizers import sgd, adam
from autograd import grad

from lds import SLDS

import transitions as trans
import observations as obs

from util import ensure_args_are_lists, ssm_pbar

class HierarchicalSLDS(SLDS):
    def __init__(self, *args, tags=(None,), lmbda=0.01, **kwargs):
        # Variance of child params around parent params
        self.lmbda = lmbda
        self.tags = tags

        # Parent model
        self.parent = SLDS(*args, **kwargs)

        # Make models for each child
        self.all_tags = np.unique(tags)
        self.children = dict()
        for tag in self.all_tags:
            ch = self.children[tag] = SLDS(*args, **kwargs)
            # hierarchical prior for transition
            ch.transitions.params = tuple(prm + np.sqrt(lmbda) * npr.randn(*prm.shape) for prm in self.parent.transitions.params) 
            # hierarchical prior for dynamics
            ch.dynamics.params = tuple(prm + np.sqrt(lmbda) * npr.randn(*prm.shape) for prm in self.parent.dynamics.params) 
            ch.emissions = self.parent.emissions  # parent and children share the same emission model            
        
        # params of hslds (define which parts of the model are hierarchical, and which parts are shared across children)
        # hierarchical transition, recurrent
        self.transitions = HierarchicalTransitions(trans.RecurrentTransitions, lmbda=self.lmbda,
                                                   K=self.parent.K, D=self.parent.D, tags = self.all_tags)
        # hierarchical dynamics, assume "AutoRegressiveObservations"
        self.dynamics = HierarchicalObservations(obs.AutoRegressiveObservations, lmbda=self.lmbda,
                                                 K=self.parent.K, D=self.parent.D, tags = self.all_tags)
        self.transitions.parent = self.parent.transitions
        self.dynamics.parent = self.parent.dynamics
        for tag in self.all_tags:
            self.transitions.children[tag] = self.children[tag].transitions
            self.dynamics.children[tag] = self.children[tag].dynamics
        
        # other params shared with parent
        self.emissions = self.parent.emissions
        self.init_state_distn = self.parent.init_state_distn
            
        self.N, self.K, self.D, self.M = self.parent.N, self.parent.K, self.parent.D, self.parent.M
        
    @ensure_args_are_lists
    def fit(self, datas, inputs=None, masks=None, tags=None,
            verbose = 2,    
            initialize=True,
            discrete_state_init_method="kmeans",
            num_init_iters=25,
            num_init_restarts=1,
            num_iters=100,        # number of EM iterations
            num_samples=1,   # number of continuous-state-samples used to approximate discrete expectations 
            continuous_optimizer="newton", # optimizer for Laplace approximation in E-step
            continuous_tolerance=1e-4,
            continuous_maxiter=100,   
            emission_optimizer="lbfgs",      # optimizer for emission model in M-step
            emission_optimizer_maxiter=100,  
            alpha=0,     
            **kwargs):

        # Initialize the model parameters
        if initialize:
            # Initialize parent SLDS using all data
            self.parent.initialize(datas, inputs, masks, tags,
                verbose=0,
                discrete_state_init_method=discrete_state_init_method,
                num_init_iters=num_init_iters,
                num_init_restarts=num_init_restarts)
            self.transitions.parent = self.parent.transitions
            self.dynamics.parent = self.parent.dynamics
            self.emissions = self.parent.emissions
            
            # Initialize each child SLDS using its own data (but each child has its own permutation/rotation of states)
            #for tag in tags:
                #self.children[tag].initialize(datas[tag],
                #                      verbose=0,
                #                       discrete_state_init_method=discrete_state_init_method,
                #                       num_init_iters=num_init_iters,
                #                       num_init_restarts=num_init_restarts)
            
            # Use the same initial parameters as parent for each child SLDS (alleviates the permutation/rotation issue)
            for tag in self.all_tags:
                # tie emission model parameters of parent and children
                self.children[tag].emissions = self.parent.emissions
                # copy transition/dynamic model parameters
                self.children[tag].transitions.params = copy.deepcopy(self.parent.transitions.params)
                self.transitions.children[tag] = self.children[tag].transitions
                self.children[tag].dynamics.params = copy.deepcopy(self.parent.dynamics.params)
                self.dynamics.children[tag] = self.children[tag].dynamics
            
        # Initialize the variational posterior
        posterior = self._make_variational_posterior("structured_meanfield", datas, inputs, masks, tags, method="laplace_em") 
            
        elbos = [self._laplace_em_elbo(posterior, datas, inputs, masks, tags)]
        pbar = ssm_pbar(num_iters, verbose, "ELBO: {:.1f}", [elbos[-1]])    
            
        # EM iterations
        for itr in pbar:
            # E-step
            # 1. Update the discrete state posterior q(z) 
            self._fit_laplace_em_discrete_state_update(posterior, datas, inputs, masks, tags, num_samples)

            # 2. Update the continuous state posterior q(x)
            self._fit_laplace_em_continuous_state_update(posterior, datas, inputs, masks, tags,
                continuous_optimizer, continuous_tolerance, continuous_maxiter)

            # M-step
            # Update parameters
            self._fit_laplace_em_params_update(posterior, datas, inputs, masks, tags,
                    emission_optimizer, emission_optimizer_maxiter, alpha)
            
        
            elbos.append(self._laplace_em_elbo(posterior, datas, inputs, masks, tags))
            if verbose == 2:
                pbar.set_description("ELBO: {:.1f}".format(elbos[-1]))
        
        return elbos, posterior
    

class _Hierarchical(object):   # this base class is mostly kept from "hierarchical.py", correcting a few errors; the two most useful functions are "log_prior" and "m_step".
    """
    Base class for hierarchical models.  Maintains a parent class and a
    bunch of children with their own perturbed parameters.
    """
    def __init__(self, base_class, *args, tags=(None,), lmbda=0.01, **kwargs):
        # Variance of child params around parent params
        self.lmbda = lmbda

        # Top-level parameters (parent)
        self.parent = base_class(*args, **kwargs)

        # Make models for each tag
        self.tags = tags
        self.children = dict()
        for tag in tags:
            ch = self.children[tag] = base_class(*args, **kwargs)
            ch.params = tuple(prm + np.sqrt(lmbda) * npr.randn(*prm.shape) for prm in self.parent.params)

    @property
    def params(self):
        prms = (self.parent.params,)
        for tag in self.tags:
            prms += (self.children[tag].params,)
        return prms

    @params.setter
    def params(self, value):
        self.parent.params = value[0]
        for tag, prms in zip(self.tags, value[1:]):
            self.children[tag].params = prms

    def permute(self, perm):
        self.parent.permute(perm)
        for tag in self.tags:
            self.children[tag].permute(perm)
            

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        self.parent.initialize(datas, inputs=inputs, masks=masks, tags=tags)
        for tag in self.tags:
            self.children[tag].params = copy.deepcopy(self.parent.params)

    def log_prior(self):
        lp = self.parent.log_prior()

        # Gaussian likelihood on each child param given parent param
        # This is the extra term in the likelihood from the mixed model extension
        for tag in self.tags:
            for pprm, cprm in zip(self.parent.params, self.children[tag].params):
                lp += np.sum(norm.logpdf(cprm, pprm, np.sqrt(self.lmbda)))
        return lp

    def m_step(self, expectations, datas, inputs, masks, tags, optimizer="adam", num_iters=25, **kwargs):
        for tag in tags:
            if not tag in self.tags:
                raise Exception("Invalid tag: ".format(tag))

        # Optimize parent and child parameters at the same time with SGD
        optimizer = dict(sgd=sgd, adam=adam)[optimizer]

        # expected log joint
        def _expected_log_joint(expectations):
            elbo = self.log_prior()
            for data, input, mask, tag, (expected_states, expected_joints, _) \
                in zip(datas, inputs, masks, tags, expectations):

                if hasattr(self.children[tag], 'log_initial_state_distn'):
                    log_pi0 = self.children[tag].log_initial_state_distn
                    elbo += np.sum(expected_states[0] * log_pi0)

                if hasattr(self.children[tag], 'log_transition_matrices'):
                    log_Ps = self.children[tag].log_transition_matrices(data, input, mask, tag)
                    elbo += np.sum(expected_joints * log_Ps)

                if hasattr(self.children[tag], 'log_likelihoods'):
                    lls = self.children[tag].log_likelihoods(data, input, mask, tag)
                    elbo += np.sum(expected_states * lls)

            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        self.params = optimizer(grad(_objective), self.params, num_iters=num_iters, **kwargs)

class HierarchicalTransitions(_Hierarchical):
    def log_transition_matrices(self, data, input, mask, tag):
        return self.children[tag].log_transition_matrices(data, input, mask, tag)  
    def transition_matrices(self, data, input, mask, tag):
        return self.children[tag].transition_matrices(data, input, mask, tag)  
    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        return self.children[tag].neg_hessian_expected_log_trans_prob(data, input, mask, tag, expected_joints)
        
class HierarchicalObservations(_Hierarchical):   # class used for the hierarchical dynamic model
    def log_likelihoods(self, data, input, mask, tag):
        return self.children[tag].log_likelihoods(data, input, mask, tag)

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        return self.children[tag].sample_x(z, xhist, input=input, tag=tag, with_noise=with_noise)

    def smooth(self, expectations, data, input, tag):
        return self.children[tag].smooth(expectations, data, input, tag)
    
    def neg_hessian_expected_log_dynamics_prob(self, Ez, x, input, x_mask, tag):
        return self.children[tag].neg_hessian_expected_log_dynamics_prob(Ez, x, input, x_mask, tag)

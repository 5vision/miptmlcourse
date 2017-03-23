
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle

from multiprocessing import Pool

def mult_random_search(f, params_mean, params_std=1., n_workers=2, batch_size=100, n_iter=50):
    """ Multiprocessing version of Random Search algorithm."""
    params_std = np.ones_like(params_mean) * params_std
    best_params = params_mean
    for _ in range(n_iter):
        pool = Pool(processes=n_workers)
        # TO BE IMPLEMENTED: random search for parameters
        yield {'results' : ys, 'best_params' : best_params}

def mult_cem(f, params_mean, params_std=1., n_workers=2, batch_size=100, n_iter=50, elite_frac=0.2):
    """ Multiprocessing version of CEM algorithm."""
    n_elite = int(np.round(batch_size * elite_frac))
    params_std = np.ones_like(params_mean) * params_std
    for _ in range(n_iter):
        pool = Pool(processes=n_workers)
        # TO BE IMPLEMENTED: CEM for search of parameters
        yield {'results' : ys, 'best_params' : best_params}


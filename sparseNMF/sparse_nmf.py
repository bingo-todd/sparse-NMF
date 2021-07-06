# -*-coding:utf-8-*-
import numpy as np


def sparseNMF(v, beta=1, cf='kl', sparsity=0, max_iter=100, conv_eps=0,
              display=False, random_seed=1, init_w=None, r=None, init_h=None,
              w_update_indices=None, h_update_indices=None, print_log=False):
    """Compute a sparse non-negative matrix factorization.

    Inputs:
    - v:  matrix to be factorized
    - beta:     beta-divergence parameter (default: 1, i.e., KL-divergence)
    - cf:       cost function type (default: 'kl'; overrides beta setting)
               'is': Itakura-Saito divergence
               'kl': Kullback-Leibler divergence
               'ed': Euclidean distance
     - sparsity: weight for the L1 sparsity penalty (default: 0)
     - max_iter: maximum number of iterations (default: 100)
     - conv_eps: threshold for early stopping (default: 0, i.e.,
            no early stopping)
     - display:  display evolution of objective function (default: False)
     - random_seed: set the random seed to the given value (default: 1;
            if equal to 0, seed is not set)
     - init_w:   initial setting for W (default: random; either init_w or
            r have to be set)
     - r:  basis functions (default: based on init_w's size; either init_w or
            r have to be set)
     - init_h:   initial setting for H (default: random)
     - w_update_indices: set of dimensions to be updated (default: all)
     - h_update_indices: set of dimensions to be updated (default: all)

    Outputs:
    - w: matrix of basis functions
    - h: matrix of activations
    - objective: objective function values throughout the iterations
    """

    def log(info):
        if print_log:
            print(info)

    m, n = v.shape

    # overwrite cost function
    if cf == 'is':  # Itakura-Saito divergence
        beta = 0
    elif cf == 'kl':  # Kullback-Leibler divergence
        beta = 1
    elif cf == 'ed':  # Euclidean distance
        beta = 2

    if random_seed > 0:
        random_seed = int(random_seed)
        log('WARNING: using non-random seed for matrix optimisation: '
            + str(random_seed))
        np.random.seed(random_seed)

    if init_w is None:  # init w
        if r is None:
            raise ValueError(
                'Number of components or initialization must be given')
        w = np.random.rand(m, r)
    else:
        ri = init_w.shape[1]
        w[:, 0:ri] = init_w   # to be tested
        if r is not None and ri < r:
            w[:, ri:r] = np.random.rand(m, r-ri)  # to be carefully checked
        else:
            r = ri

    if init_h is None:  # init h
        h = np.random.rand(r, n)
    else:
        h = init_h

    # specify which colums of w will be updated
    if w_update_indices is None:
        w_update_indices = np.ones(r, dtype=np.bool)
    if h_update_indices is None:
        h_update_indices = np.ones(r, dtype=np.bool)

    # Normalize the columns of W and rescale H accordingly
    wn = np.sqrt(np.sum(w**2, axis=0))
    w = w/wn[np.newaxis, :]
    # h = h*wn[:, np.newaxis]

    # Internal parameters
    flr = 1e-9
    lamb = np.dot(w, h)  # used to be lambda (but lambda is a reserved
    #                    keyword in Python)
    lamb[lamb < flr] = flr
    last_cost = np.inf

    objective_div = np.zeros((max_iter,))
    objective_cost = np.zeros((max_iter,))
    div_beta = beta
    h_ind = h_update_indices.flatten()
    w_ind = w_update_indices.flatten()
    update_h = np.sum(h_ind) > 0
    update_w = np.sum(w_ind) > 0

    log(f'INFO: Performing sparse NMF with beta-divergence, beta={div_beta}')
    for it in range(max_iter):
        # updates H
        if update_h > 0:
            if div_beta == 1:
                dph = np.sum(w[:,  h_ind], axis=0)[:, np.newaxis]+sparsity
                dmh = np.dot(w[:, h_ind].T, v/lamb)
            elif div_beta == 2:
                dph = np.dot(w[:, h_ind].T, lamb)+sparsity
                dmh = np.dot(w[:, h_ind].T, v)
            else:
                dph = np.dot(w[:, h_ind].T, lamb**(div_beta-1))+sparsity
                dmh = np.dot(w[:, h_ind].T, v*lamb**(div_beta-2))
            #
            dph[dph < flr] = flr
            h[h_ind, :] = h[h_ind, :]*dmh/dph
            lamb = np.dot(w, h)
            lamb[lamb < flr] = flr

        # updates W
        if update_w > 0:
            if div_beta == 1:
                dpw = (np.sum(h[w_ind, :], axis=1)[np.newaxis, :]
                       + np.sum(
                           np.dot(v/lamb, h[w_ind, :].T)*w[:, w_ind],
                           axis=0)[np.newaxis, :]
                       * w[:, w_ind])
                dmw = (np.dot(v/lamb, h[w_ind, :].T)
                       + np.sum(
                           np.sum(h[w_ind, :], axis=1)[np.newaxis, :]
                           * w[:, w_ind], axis=0)[np.newaxis, :]
                       * w[:, w_ind])
            elif div_beta == 2:
                dpw = (np.dot(lamb, h[w_ind, :].T)
                       + np.sum(
                           np.dot(v, h[w_ind, :].T)*w[:, w_ind],
                           axis=0)[np.newaxis, :]
                       * w[:, w_ind])
                dmw = (np.dot(v, h[w_ind, :].T)
                       + np.sum(
                           np.dot(lamb, h[w_ind, :].T)*w[:, w_ind],
                           axis=0)[np.newaxis, :]
                       * w[:, w_ind])
            else:
                dpw = (np.dot(lamb**(div_beta-1), h[w_ind, :].T)
                       + np.sum(
                           np.dot(v*lamb**(div_beta-2), h[w_ind, :].T)
                           * w[:, w_ind],
                           axis=0)[np.newaxis, :]
                       * w[:, w_ind])
                dmw = (np.dot(v*lamb**(div_beta-2), h[w_ind, :].T)
                       + np.sum(
                           np.dot(lamb**(div_beta-1), h[w_ind, :].T)
                           * w[:, w_ind],
                           axis=0)[np.newaxis, :]
                       * w[:, w_ind])
            dpw[dpw < flr] = flr
            w[:, w_ind] = w[:, w_ind]*dmw/dpw
            # Normalize the columns of W
            w = w/np.sqrt(np.sum(w**2, axis=0, keepdims=True))
            lamb = np.dot(w, h)
            lamb[lamb < flr] = flr

        # compute the objective function
        if div_beta == 1:
            div = np.sum(v*np.log(v/lamb+1e-20)-v+lamb)
        elif div_beta == 2:
            div = np.sum((v-lamb)**2)
        elif div_beta == 0:
            div = np.sum(v/lamb-np.log(v/lamb+1e-20)-1)
        else:
            div = (np.sum(v**div_beta
                          + (div_beta-1)*lamb**div_beta
                          - div_beta*v*lamb**(div_beta-1))
                   / (div_beta*(div_beta-1)))
        cost = div + np.sum(sparsity*h)

        objective_div[it] = div
        objective_cost[it] = cost

        log(f'iteration {it}: div={div}, cost={cost}')
        # convergence check
        if it > 0 and conv_eps > 0:
            e = np.abs(cost - last_cost)/last_cost
            if e < conv_eps:
                log('Convergence reached, aborting iteration')
                objective_div = objective_div[:it]
                objective_cost = objective_cost[:it]
                break  # exit the loop
        last_cost = cost
    last_cost = cost
    return w, h, objective_div, objective_cost

"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    mu = np.zeros((n, K))

    for i in range(n):
        tiled_x = np.tile(X[i, :], (K, 1))
        mask = tiled_x != 0.0
        mu += (np.tile(post[i,:],(d,1)).T*tiled_x*mask.astype(int))

    return mu

    # def normal(x, mu, var):
    #     return 1 / ((2 * var * np.pi) ** (d / 2)) * np.exp(-1 / (2 * var) * ((x - mu) ** 2).sum(axis=1))
    #
    # for i in range(n):
    #     tiled_x = np.tile(X[i, :], (K, 1))
    #     N = mixture.p * normal(tiled_x, mixture.mu, mixture.var)
    #     for k in range(K):
    #         post[i, k] = N[k] / np.sum(N)
    #         L[i,] = np.sum(N)
    #
    # ll = np.sum(np.log(L))
    # return post, ll






def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    numerator = np.zeros((n, K))
    denominator = np.zeros((n, K))

    for i in range(n):
        break


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_ll = None
    new_ll = None
    while (old_ll is None or (new_ll - old_ll) >= (1e-6 * abs(new_ll))):
        old_ll = new_ll
        post, new_ll = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, new_ll


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError

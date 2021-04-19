"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    L = np.zeros((n,))

    def normal(x, mu, var):
        return 1 / ((2 * var * np.pi) ** (d / 2)) * np.exp(-1 / (2 * var) * ((x - mu) ** 2).sum(axis=1))

    for i in range(n):
        tiled_x = np.tile(X[i, :], (K, 1))
        N = mixture.p * normal(tiled_x, mixture.mu, mixture.var)
        for k in range(K):
            post[i, k] = N[k] / np.sum(N)
            L[i,] = np.sum(N)

    ll = np.sum(np.log(L))
    return post, ll

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

    n_hat = post.sum(axis=0)
    p = n_hat / n

    cost = 0
    mu = np.zeros((K, d))
    var = np.zeros(K)

    for j in range(K):
        mu[j, :] = post[:, j] @ X / n_hat[j]
        sse = ((mu[j] - X)**2).sum(axis=1) @ post[:, j]
        cost += sse
        var[j] = sse / (d * n_hat[j])

    return GaussianMixture(mu, var, p)

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
    raise NotImplementedError

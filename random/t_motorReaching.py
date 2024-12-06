#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:50:00 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jxr
import matplotlib.pyplot as plt
from tqdm import trange

#%%
#@title Model hyperparameters

# `scale` parameter controls the size of the covariances.
# `decay` parameter controls the smoothness of the Wishart process.
# `truncation_tol` will control how many basis functions are kept.
scale = 0.1
decay = 2.0
truncation_tol = 1e-6

# `num_dims` is the number of dimensions that the reach targets span.
# `extra_dims` is the degrees-of-freedom parameter of the Wishart distribution.
# Usually it seems helpful to set `extra_dims` equal to 1 or 2.
num_dims = 2
extra_dims = 2

# `num_reaches` is the number of datapoints we observe. I assume that the
# reaches are equispaced on [0, 2*pi) with only one trial per condition.
num_reaches = 300

# Compute and plot the decaying eigenvalue spectrum. Each eigenvalue is equal to
# the variance of the prior at that Fourier basis frequency.
eigvals = [scale]
while eigvals[-1] > truncation_tol:
    eigvals.append(scale * jnp.exp(-len(eigvals) * decay))
eigvals = jnp.array(eigvals)
num_freq = len(eigvals)

# Plot the eigenvalue spectrum
plt.plot(eigvals, '.-')
plt.xlabel("Frequency")
plt.ylabel("Eigenvalue")

#%%
#@title Function to evaluate the weighted Fourier basis
def eval_basis(params, eigvals, theta):
    """
    Evaluate the Fourier basis functions.

    Args:
        params: (2 * num_frequencies)-vector of basis function weights.
        eigvals: (num_frequencies)-vector of variance for each basis function.
        theta: vector specifying where to evaluate the basis functions
    """
    w_sin, w_cos = jnp.array_split(params, 2)
    freq = jnp.arange(1, num_freq + 1)
    return jnp.sqrt(eigvals)[:, None] * (
        w_sin[:, None] * jnp.sin(theta[None, :] * freq[:, None]) +
        w_cos[:, None] * jnp.cos(theta[None, :] * freq[:, None])
    )

#%%
#@title Demonstration of samples from the Gaussian process prior.

keys = [jxr.PRNGKey(i) for i in range(1000)]

fig, ax = plt.subplots(1, 1, sharey=True)
theta = jnp.linspace(0, 2 * jnp.pi, num_reaches)

for sample in range(5):
    params = jxr.normal(keys.pop(0), shape=(2 * num_freq,)) #14
    basis_s = eval_basis(params, eigvals, theta) #7 x 300
    ax.plot(jnp.sum(basis_s, axis=0), 'k', alpha=.5)

ax.set_title("Samples from the GP prior")
ax.set_xlabel("Reach Angle (theta)")

#%%
#@title Function to evaluate the Wishart process

# This uses some jax magic to stack the Gaussian processes into
# (num_dims x num_dims) shape.
eval_basis_matrix = jax.vmap(
    jax.vmap(eval_basis, in_axes=(0, None, None)), in_axes=(0, None, None)
)

def eval_wishart(params, eigvals, theta):
    """
    Evaluates the Wishart process at a point in parameter space specified by
    `params` array.

    Args:
        params: (num_dims + extra_dims x num_dims x 2 * num_frequencies) array of weights.
        eigvals: (num_frequencies)-vector of variance for each basis function.
        theta: vector specifying where to evaluate the basis functions.

    Returns:
        covs: len(theta) x num_dims x num_dims array of covariances.
    """
    U = jnp.sum(eval_basis_matrix(params, eigvals, theta), axis=2) #4 x 2 x 7 x 300 -> 4 x 2 x 300
    return jnp.einsum("kit,kjt->tij", U, U) #300 x 2 x 2

def psd_transform(S, x):
    """
    Multiply `x` by the square root of a positive definite matrix `S`.
    """
    v, U = jnp.linalg.eigh(S)
    v = jnp.maximum(v, 0)
    return U @ jnp.diag(jnp.sqrt(v)) @ U.T @ x


#%%
#@title Sample a ground truth set of covariances

# Sample some "true" covariances to simulate data
true_params = jxr.normal(
    keys.pop(0), shape=(num_dims + extra_dims, num_dims, 2 * num_freq)
)
true_covs = eval_wishart(true_params, eigvals, theta)
print(true_covs.shape)

# Visualize the ground truth covariances
for i in range(0, num_reaches, num_reaches // 30):
    _x, _y = psd_transform(true_covs[i], jnp.stack((jnp.cos(theta), jnp.sin(theta))))
    plt.plot(_x + 10 * jnp.cos(theta[i]), _y + 10 * jnp.sin(theta[i]), "-k", alpha=.5)
    
#%%
#@title Simulate noisy data from the ground truth (one trial per condition)

noisy_data = jax.vmap(psd_transform, in_axes=(0, 0))(
    true_covs,
    jxr.normal(keys.pop(0), shape=(len(theta), num_dims))
)

# Visualize the ground truth covariances
for i in range(0, num_reaches, num_reaches // 30):
    _x, _y = psd_transform(true_covs[i], jnp.stack((jnp.cos(theta), jnp.sin(theta))))
    plt.plot(
        _x + 10 * jnp.cos(theta[i]),
        _y + 10 * jnp.sin(theta[i]),
        "-k", alpha=.5
    )
    plt.scatter(
        noisy_data[i, 0] + 10 * jnp.cos(theta[i]),
        noisy_data[i, 1] + 10 * jnp.sin(theta[i]),
        color='r'
    )

#%%
#@title Sample initial estimate of covariances

# Sample another set of covariances, which will initialize the optimizer
init_est_params = jxr.normal(
    keys.pop(0), shape=(num_dims + extra_dims, num_dims, 2 * num_freq)
)
init_est_covs = eval_wishart(init_est_params, eigvals, theta)

# Visualize the ground truth covariances
for i in range(0, num_reaches, num_reaches // 30):
    _x, _y = psd_transform(true_covs[i], jnp.stack((jnp.cos(theta), jnp.sin(theta))))
    plt.plot(
        _x + 10 * jnp.cos(theta[i]),
        _y + 10 * jnp.sin(theta[i]),
        "-k", alpha=.75, label="ground truth"
    )
    _x, _y = psd_transform(init_est_covs[i], jnp.stack((jnp.cos(theta), jnp.sin(theta))))
    plt.plot(
        _x + 10 * jnp.cos(theta[i]),
        _y + 10 * jnp.sin(theta[i]),
        "-r", alpha=.5, label="initialization"
    )
    if i == 0:
        plt.legend()
        
#%%
#@title Function to compute the log posterior (up to a constant)

def log_unnrm_posterior(params, data, eigvals, theta):

    # Right now, we are assuming reach data is mean-subtracted per condition.
    means = jnp.zeros((len(theta), num_dims))

    # Evaluate the wishart process to get the covariances.
    covs = eval_wishart(params, eigvals, theta)

    # Compute the log likelihood and prior.
    log_likelihood = jnp.sum(
        jax.scipy.stats.multivariate_normal.logpdf(data, means, covs)
    )
    log_prior = jnp.sum(
        jax.scipy.stats.norm.logpdf(params.ravel())
    )
    return log_likelihood + log_prior


# Create a function that automatically computes the objective + gradient
objective = jax.jit(jax.value_and_grad(log_unnrm_posterior))

log_unnrm_posterior(true_params, noisy_data, eigvals, theta)

#%%
#@title Fit using gradient ascent on the log posterior

# Sample another set of covariances, which will initialize the optimizer
est_params = jnp.copy(init_est_params)

# This needs to be tuned
learning_rate = 0.0001
num_iterations = 1000

# Run gradient ascent
obj_hist = []
for i in trange(num_iterations):
    val, grad = objective(est_params, noisy_data, eigvals, theta)
    est_params = est_params + learning_rate * grad
    obj_hist.append(val)

# Plot objective
plt.plot(obj_hist)
plt.xlabel("iterations")
plt.ylabel("log posterior")

#%%
#@title Visualize optimized covariances

est_covs = eval_wishart(est_params, eigvals, theta)

for i in range(0, num_reaches, num_reaches // 30):
    _x, _y = psd_transform(true_covs[i], jnp.stack((jnp.cos(theta), jnp.sin(theta))))
    plt.plot(
        _x + 10 * jnp.cos(theta[i]),
        _y + 10 * jnp.sin(theta[i]),
        "-k", alpha=.75, label=("ground truth" if i == 0 else None)
    )
    _x, _y = psd_transform(est_covs[i], jnp.stack((jnp.cos(theta), jnp.sin(theta))))
    plt.plot(
        _x + 10 * jnp.cos(theta[i]),
        _y + 10 * jnp.sin(theta[i]),
        "-b", alpha=.5, label=("estimated" if i == 0 else None)
    )
plt.scatter(
    noisy_data[:, 0] + 10 * jnp.cos(theta),
    noisy_data[:, 1] + 10 * jnp.sin(theta),
    color='r', label="data", lw=0, s=5
)
plt.legend()

#%%








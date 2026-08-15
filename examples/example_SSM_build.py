#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 23:41:43 2026

@author: jacopo
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import jax
import geossm

# %% import and check the version
# pip install -e geossm


print("Version: ", geossm.__version__)
print("Load from: ", geossm.__file__)

# Import the StateSpaceModel class from geossm.ssm
if geossm.__file__:
    from geossm.ssm import StateSpaceModel as ssm


# %% Create the State Space Model (SSM)
# Type = linear time-invariant SSM

# create the model parameters
p = 10
q = 6
b = 3
T = 10

# markovian matrix
F = 0.85 * np.eye(q)

# mapping matrix
H = np.hstack((np.ones((p, 1)), np.random.binomial(1, 0.5, size=(p, q - 1))))

# measueremtent error covaraince matrix
R = 0.2 * np.eye(p)

# innovetion covariance matrix
Q = 0.5 * np.eye(q)

# regression design matrix
Xbeta = np.random.normal(0, 1, size=(p, b, T))
beta = np.ones(b)

# %% Build the model
# default backend = auto ('cpu' or 'gpu')
model = ssm(H, R, F, Q, Xbeta=Xbeta, beta=beta, x0=None, Sigma0=None, dtype=np.float32)

# %% print the model summary
print(model)
print(model.summary())

# %% Save the model using pickle
filename = "model.pkl"

with open(filename, "wb") as file:
    pickle.dump(model, file)

# load the model
with open(filename, "rb") as file:
    mymodel = pickle.load(file)
    print(mymodel)

# %% Get and Set model attibute

# get model attribute
model.R.shape
model.x0.shape
model.Sigma0.shape
print(model.p)
print(model.q)
print(model.q)
print(model.T)
print(model.R.device)
model.shape

# set model attribute
F = 0.8 * np.eye(q)
model.set(F=F)
print(model.F)  # F is stored as its diagonal (1D) directly
print(model.F.device)
# %% Simulate the model

# Use the default Xbeta (see the ssm() for details)
y_sim, x_sim, tdelta = model.sim(seed=1234)
print("Simulate response y:", y_sim.shape)
print("Simulate stete x:", x_sim.shape)
print("Computation time tDelta (s):", tdelta)


# %% Simulate with some update paramites Xbeta

Xbeta = np.random.normal(1, 2, size=(p, b, 100))
y_sim, x_sim, stats, tdelta = model.sim(seed=1234, Xbeta=Xbeta)

# Simulate without stats
# y_sim, x_sim, stats, tdelta = model.sim(seed=1234, Xbeta=Xbeta, stats = False)

print("Simulate response y:", y_sim.shape)
print("Simulate stete x:", x_sim.shape)
print("Computation time tDelta (s):", tdelta)
print(f"backend: {y_sim.device},{x_sim.device}")

# plot one time-series
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_sim[0, :], label="Simulated observation (y)")
ax.plot(x_sim[0, :], ":", label="Simulated state (x)")
ax.set_title("Simulated Time Series")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend()
plt.show()

# %% Compare the CPU and GPU backend computation time

# Only benchmark backends that are actually available on this machine.
# jax.devices("gpu") raises a RuntimeError (rather than returning an empty
# list) when no GPU platform is registered, so this must be caught.
try:
    has_gpu = bool(jax.devices("gpu"))
except RuntimeError:
    has_gpu = False

backends = ["cpu", "gpu"] if has_gpu else ["cpu"]
if not has_gpu:
    print("No GPU device found: skipping the GPU backend in the timing comparison.")

# number of Monte Carlo repetitions per configuration, used to average out
# timing noise (JIT/dispatch overhead, OS scheduling, etc.)
n_mc = 10
rng = np.random.default_rng(42)


def make_matrices(p, q, b, T, rng):
    """Build a random set of SSM matrices for a given (p, q, b, T)."""
    F = 0.85 * np.eye(q)
    H = np.hstack((np.ones((p, 1)), rng.binomial(1, 0.5, size=(p, q - 1))))
    R = 0.2 * np.eye(p)
    Q = 0.5 * np.eye(q)
    Xbeta = rng.normal(0, 1, size=(p, b, T))
    beta = np.ones(b)
    return F, H, R, Q, Xbeta, beta


def time_model(model, n_mc, seed0=1234):
    """Run `n_mc` simulations and return the (mean, std, raw) computation times."""
    times = np.array(
        [model.sim(seed=seed0 + i, stats=False)[-1] for i in range(n_mc)]
    )
    return times.mean(), times.std(), times


# Baseline dimensions, held fixed while sweeping one dimension at a time
p_base, q_base, b_base, T_base = 200, 60, 3, 1000

sweeps = {
    "T": {"values": [100, 200, 500, 1000, 2000, 5000], "fixed": {"p": p_base, "q": q_base, "b": b_base}},
    "p": {"values": [50, 100, 200, 500, 1000], "fixed": {"q": q_base, "b": b_base, "T": T_base}},
    "q": {"values": [10, 20, 60, 120, 250], "fixed": {"p": p_base, "b": b_base, "T": T_base}},
}

records = []
for sweep_name, cfg in sweeps.items():
    for value in cfg["values"]:
        dims = dict(cfg["fixed"])
        dims[sweep_name] = value

        # Same data/matrices for every backend -> a fair apples-to-apples comparison
        F, H, R, Q, Xbeta, beta = make_matrices(dims["p"], dims["q"], dims["b"], dims["T"], rng)

        for backend in backends:
            model = ssm(H, R, F, Q, Xbeta=Xbeta, beta=beta, backend=backend)
            tmean, tstd, _ = time_model(model, n_mc)

            records.append(
                {
                    "sweep": sweep_name,
                    "value": value,
                    "backend": backend,
                    "p": dims["p"],
                    "q": dims["q"],
                    "b": dims["b"],
                    "T": dims["T"],
                    "tsim_mean": tmean,
                    "tsim_std": tstd,
                }
            )
            print(
                f"[sweep={sweep_name:<1s}] backend={backend:>3s} "
                f"{sweep_name}={value:<5d} tsim_mean(s)={tmean:.4f} tsim_std(s)={tstd:.4f} "
                f"(n_mc={n_mc})"
            )

timing_df = pd.DataFrame.from_records(records)

# %% Plot runtime vs T, p and q for each backend

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, sweep_name in zip(axes, ["T", "p", "q"]):
    sub = timing_df[timing_df["sweep"] == sweep_name]
    for backend in backends:
        s = sub[sub["backend"] == backend].sort_values("value")
        ax.errorbar(
            s["value"],
            s["tsim_mean"],
            yerr=s["tsim_std"],
            marker="o",
            capsize=3,
            label=backend.upper(),
        )
    ax.set_xlabel(sweep_name)
    ax.set_ylabel("Simulation time (s)")
    ax.set_title(f"Runtime vs {sweep_name}")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle(f"CPU vs GPU simulation time (mean ± std over {n_mc} MC repetitions)")
fig.tight_layout()
plt.show()


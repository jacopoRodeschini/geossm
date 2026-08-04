#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacopo
"""

import numpy as np
import matplotlib.pyplot as plt
import jax
from geossm.ssm import StateSpaceModel as ssm

# %% create the ssm model
p = 10
q = 2
b = 3
T = 50

# markovian matrix
F = 0.85 * np.eye(q)

# mapping matrix
H = np.hstack((np.ones((p, 1)), np.random.binomial(1, 0.5, size=(p, q - 1))))

# measueremtent error covaraince matrix
R = 8 * np.eye(p)

# innovetion covariance matrix
Q = 10 * np.eye(q)

# regression design matrix
Xbeta = np.random.normal(0, 1, size=(p, b, T))
beta = np.ones(b)

# %% Build the model
model = ssm(H, R, F, Q, Xbeta=Xbeta, beta=beta, x0=None, Sigma0=None, dtype=np.float32)

print(model)

# %% Simulate the model
y_sim, x_sim, stats, tdelta = model.sim(seed=1234)

print("Simulate response y:", y_sim.shape)
print("Simulate stete x:", x_sim.shape)
print("runtime", tdelta)

# plot one time-series
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_sim[0, :], label="Simulated observation (y)")
ax.plot(x_sim[0, :T], ":", label="Simulated state (x)")
ax.set_title("Simulated Time Series")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend()
plt.show()

# %% Simulate with dirrent beta

# regression design matrix
Xbeta = np.random.normal(1, 2, size=(p, b, 100))
beta = 2 * np.ones(b)

y_sim, x_sim, stats, tdelta = model.sim(seed=1234, Xbeta=Xbeta, beta=beta)

T = model.T

print("Simulate response y:", y_sim.shape)
print("Simulate stete x:", x_sim.shape)
print("runtime", tdelta)

# plot one time-series
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_sim[0, :], label="Simulated observation (y)")
ax.plot(x_sim[0, :T], ":", label="Simulated state (x)")
ax.set_title("Simulated Time Series")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend()
plt.show()


# %% Estimate the model (by filtering the data)

results = model.filter(y_sim)
print(results)

# print the summary of the results
print(results)  # = print(results.summary())

# %% Coverage probability of the state

lower, upper = results.conf_int_state(alpha=0.05, which="filtered")
t = np.arange(T)

plt.figure()
plt.plot(t, x_sim[0, :T], label="Simulated state (x)")
plt.plot(t, results.x_filtered[0, 1:], ":", label="Filtered state (x_filtered)")
plt.fill_between(
    t, lower[0][1:], upper[0][1:], alpha=0.3, label="95% Confidence Interval"
)
plt.grid()
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Filtered State with 95% Confidence Interval")
plt.legend()
plt.show()

# compute the coverage probability
inner = (x_sim[0, :T] >= lower[0, 1:]) & (x_sim[0, :T] <= upper[0, 1:])
coverage_probability = np.mean(inner)
print("Coverage Probability of the Filtered State:", coverage_probability)


# %% The residual array are all numpy array

# Get residuals
res = results.residuals

# Plot the grouped by boxplot (firs 50 days)
plt.figure(figsize=(8, 6))
plt.boxplot(res[:, :50], label=[f"${i}$" for i in range(res.shape[1])][:50])
plt.title("Boxplot of Residuals by Time")
plt.xlabel("Time")
plt.ylabel("Residuals")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid()
plt.show()

# %% Get the goodness of fit
# get the mse
mse_global = results.mse("global")
mse_space = results.mse("space")
mse_time = results.mse("time")

print("MSE global:", mse_global)
print("MSE space:", mse_space.mean())
print("MSE time:", mse_time.mean())

# plot the mse in time
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(mse_time,'x', linestyle='--', label="MSE time")
ax.set_xlabel("Time")
ax.set_ylabel("MSE")
ax.set_title("Mean Squared Error Over Time")
ax.legend()
ax.grid()
plt.show()

# %% Get the prediction and confidence interval

# get the predicted values
y_hat = results.y_hat

# get the confidence interval
lower, upper = results.conf_int_y(alpha=0.05, prediction=True)


t = np.arange(y_hat.shape[1])

plt.figure()
plt.plot(t, y_hat[0])
plt.plot(t, y_sim[0], "x")
plt.fill_between(t, lower[0], upper[0], alpha=0.3)
plt.grid()
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Prediction with 95% Confidence Interval")
plt.show()

# %% Get the residual diagnostics

stats = results.diagnostics()

# print the diagnostics
print("Jarque-Bera test statistic:", stats["jb"])
print("Jarque-Bera p-value:", stats["jb_pvalue"])
print("Omnibus test statistic:", stats["omni"])
print("Omnibus test p-value:", stats["omni_pvalue"])


# %% Get dictionaly of the resutls

results_dict = results.as_dict()

# %% Get the coverage probability

cov_prov_global = results.coverage_probability(which="global")
cov_prov_space = results.coverage_probability(which="space")
cov_prov_time = results.coverage_probability(which="time")

print("Coverage probability global:", cov_prov_global)
print("Coverage probability space:", cov_prov_space.mean())
print("Coverage probability time:", cov_prov_time.mean())


# %% Example of filtering

# create the model
p = 1
q = 1
b = 3
T = 100

F = 0.90 * np.eye(q)
H = np.hstack((np.ones((p, 1)), np.random.binomial(1, 0.5, size=(p, q - 1))))
R = 2 * np.eye(p)
Q = 6 * np.eye(q)

Xbeta = np.random.normal(0, 1, size=(p, b, T))
beta = np.ones(b)

model = ssm(
    H, R, F, Q, Xbeta=Xbeta, beta=beta, x0=None, Sigma0=None, dtype=jax.numpy.float32
)

print(model.summary())

# simulate the data
y_sim, x_sim, tdelta = model.sim(seed=1234)

# filter the state
results = model.filter(y_sim)

print("Filtered state x_t:", results.x_filtered.shape)
print("Filtered state covariance P_t:", results.P_filtered.shape)
print("Kalman gain K:", results.K.shape)
print("Predicted state x_t_1:", results.x_pred.shape)
print("Predicted state covariance P_t_1:", results.P_pred.shape)
print("Predicted state invP_t_1:", results.invP_pred.shape)
print("Log-likelihood logL:", results.llf)
print("Computation time tDelta (s):", results.time_filter)

x_t = results.x_filtered[:, 1:]

# plot one time-series
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_sim[0, :], label="Simulated observation (y)")
ax.plot(x_sim[0, :T], label="Simulated state (x_sim)")
ax.plot(x_t[0, :], ":", label="Filtered state (x_t)")
ax.set_title("Filtered Time Series")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend()
plt.show()

# print the result summary
print(results)

# %% Example of filtering under increasing noise
p = 2
q = 2
b = 3
T = 100

F = 0.90 * np.eye(q)
H = np.hstack((np.ones((p, 1)), np.random.binomial(1, 0.5, size=(p, q - 1))))
Q = 6 * np.eye(q)

Xbeta = np.random.normal(0, 1, size=(p, b, T))
beta = np.ones(b)

rmse = []
d = 10
num = 20
for sigma2e in np.linspace(1, d, num=num):
    R = sigma2e * np.eye(p)
    model = ssm(H, R, F, Q, Xbeta=Xbeta, beta=beta)

    # simulate the data
    y_sim, x_sim, stats, tdelta = model.sim(seed=1234)

    # filter the state
    res = model.filter(y_sim)

    # compute the state rmse
    rmse.append(np.sqrt(np.mean((x_sim[:, :T] - res.x_filtered[:, 1:]) ** 2)))

# plot the rmse
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(rmse, marker="o")
ax.set_title("State RMSE vs Measurement Noise Variance")
ax.set_xlabel("Measurement Noise Variance (sigma^2)")
ax.set_ylabel("RMSE")
ax.set_xticks(range(len(np.linspace(1, d, num=num))))
ax.set_xticklabels([f"{sigma2:.2f}" for sigma2 in np.linspace(1, d, num=num)])
plt.show()


# %% Estimated coverage probability under increasing level of noise

# increasing noise variance
p = 100
q = 10
b = 3
T = 100

F = 0.90 * np.eye(q)
H = np.hstack((np.ones((p, 1)), np.random.binomial(1, 0.5, size=(p, q - 1))))
Q = 1 * np.eye(q)

Xbeta = np.random.normal(0, 1, size=(p, b, T))
beta = np.ones(b)

cov_prob = []
d = 10
num = 20
domain = np.linspace(1, d, num=num)
for sigma2e in domain:
    R = sigma2e * np.eye(p)
    model = ssm(H, R, F, Q, Xbeta=Xbeta, beta=beta)

    # simulate the data
    y_sim, x_sim, stats, tdelta = model.sim(seed=1234)

    # filter the state
    res = model.filter(y_sim)

    # compute the state rmse
    cov_prob.append(res.coverage_probability(which="global"))


snr = np.mean(Q.diagonal()) / domain

# plot the rmse
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(cov_prob, marker="o")
ax.set_title("State RMSE vs Measurement Noise Variance")
ax.set_xlabel("Measurement Noise Variance (sigma^2)")
ax.set_ylabel("RMSE")
ax.set_xticks(range(len(np.linspace(1, d, num=num))))
ax.set_xticklabels([f"{lam:.2f}" for lam in snr])
plt.show()


# %% Compare filterin using CPU and GPU backend

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
n_mc = 20
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
    y_sim, x_sim, stats, tdelta = model.sim(seed=seed0, stats=False)
    
    times = np.array(
        [model.filter(y_sim).time_total for i in range(n_mc)]
    )
    return times.mean(), times.std(), times


# Baseline dimensions, held fixed while sweeping one dimension at a time
p_base, q_base, b_base, T_base = 200, 60, 3, 1000

sweeps = {
    "T": {"values": [100, 200, 500, 1000, 2000, 5000], "fixed": {"p": p_base, "q": q_base, "b": b_base}},
    "p": {"values": [50, 100, 200, 500, 1000], "fixed": {"q": q_base, "b": b_base, "T": T_base}},
    "q": {"values": [50, 100, 200, 500, 1000], "fixed": {"p": p_base, "b": b_base, "T": T_base}},
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
            tmean, tstd, times = time_model(model, n_mc)

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
                    "tsim_median": np.median(times),
                }
            )
            print(
                f"[sweep={sweep_name:<1s}] backend={backend:>3s} "
                f"{sweep_name}={value:<5d} tsim_mean(s)={tmean:.4f} tsim_std(s)={tstd:.4f} "
                f"(n_mc={n_mc})"
            )

import pandas as pd
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


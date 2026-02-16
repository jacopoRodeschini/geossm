#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Feb.  16 23:41:43 2026

@author: jacopo
"""
import numpy as np
import matplotlib.pyplot as plt
import jax

from geossm.ssm import StateSpaceModel as ssm

# %% create the ssm model
p = 10
q = 6
b = 3
T = 100

# markovian matrix
F = 0.85 * np.eye(q)

# mapping matrix
H = np.hstack((np.ones((p, 1)), np.random.binomial(1, 0.5, size=(p, q-1))))

# measueremtent error covaraince matrix
R = 0.2 * np.eye(p)

# innovetion covariance matrix
Q = 0.5 * np.eye(q)

# regression design matrix
Xbeta = np.random.normal(0, 1, size=(p, b, T))
beta = np.ones(b)

# %% Build the model
model = ssm(H, R, F, Q, Xbeta=Xbeta, beta=beta,
            x0=None, Sigma0=None, dtype=np.float32)

print(model)

# %% Simulate the model

Xbeta = np.random.normal(1, 2, size=(p, b, T))
y_sim, x_sim, tdelta = model.sim(seed=1234, Xbeta=Xbeta)

print("Simulate response y:", y_sim.shape)
print("Simulate stete x:", x_sim.shape)
print("runtime", tdelta)

# plot one time-series
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_sim[0, :], label='Simulated observation (y)')
ax.plot(x_sim[0, :], ':', label='Simulated state (x)')
ax.set_title('Simulated Time Series')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
plt.show()


# %% Estimate the model (estimate the state equation)

results = model.filter(y_sim)


print(results.summary())



# %% The residual array are all numpy array 


# get residuals
res = results.residuals

# get the mse
mse_global= results.mse('global')
mse_space= results.mse('space')
mse_time= results.mse('time')

# plot the mse in time
plt.plot(mse_time)

# %% Get the prediction

y_hat = results.y_hat

lower, upper = results.conf_int_y(alpha=0.05)


t = np.arange(y_hat.shape[1])

plt.figure()
plt.plot(t, y_hat[0])
plt.plot(t, y_sim[0],'x')
plt.fill_between(t, lower[0], upper[0], alpha=0.3)

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Prediction with 95% Confidence Interval")

plt.show()

# %% Get the residual diagnostics

stats = results.diagnostics()

# %% Get dictionaly of the resutls

results_dict = results.as_dict()

# %% Get the coverage probability

cov_prov_space = results.coverage_probability(which='space')








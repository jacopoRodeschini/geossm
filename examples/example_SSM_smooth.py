#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Feb.  17, 2026

@author: jacopo
"""
import numpy as np
import matplotlib.pyplot as plt
import jax

from geossm.ssm import StateSpaceModel as ssm
from geossm.utils import KeyStream

# %% Example of smoothing the state of a SSM

# create the model
p = 1
q = 1
b = 3
T = 100

F = 0.90 * np.eye(q)
H = np.hstack((np.ones((p, 1)), np.random.binomial(1, 0.5, size=(p, q-1))))
R = 0.2 * np.eye(p)
Q = 0.6 * np.eye(q)

Xbeta = np.random.normal(0, 1, size=(p, b, T))
beta = np.ones(b)

model = ssm(H, R, F, Q, Xbeta=Xbeta, beta=beta)

print(model.summary(print_output='short'))

# simulate the data
y_sim, x_sim, tdelta = model.sim(seed=1234)

# smooth the state
results = model.smoother(y_sim)

# %%

results.summary()

# %%
# plot one time-series
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_sim[0, :], label='Simulated observation (y)')
ax.plot(x_sim[0, :], label='Simulated state (x_sim)')
ax.plot(results.x_smoothed[0, :], 'g:', label='Smoothed state (x_t)')
ax.plot(results.x_filtered[0, :], 'r:', label='Filtered state (x_t)')
ax.set_title('Smoothed Time Series')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
plt.show()


# %% Smoothing performance under increasing noise variance
p = 1
q = 1
b = 3
T = 100

F = 0.90 * np.eye(q)
H = np.hstack((np.ones((p, 1)), np.random.binomial(1, 0.5, size=(p, q-1))))
Q = 0.6 * np.eye(q)

Xbeta = np.random.normal(0, 1, size=(p, b, T))
beta = np.ones(b)

rmse = []
d = 10
num = 20
for sigma2 in np.linspace(0.1, 10, num=num):
    R = sigma2 * np.eye(p)

    model = ssm(H, R, F, Q, Xbeta=Xbeta, beta=beta)

    # simulate the data
    y_sim, x_sim = model.sim(seed=1234)

    # smooth the state
    x_sm, P_sm, P_T_1_sm, tDelta = model.smoother(y_sim)

    # compute the rmse
    rmse.append(np.sqrt(np.mean((x_sim - x_sm)**2)))

# plot the rmse
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(rmse, marker='o')
ax.set_title('RMSE vs Measurement Noise Variance')
ax.set_xlabel('Measurement Noise Variance (sigma^2)')
ax.set_ylabel('RMSE')
ax.set_xticks(range(len(np.linspace(0.1, d, num=num))))
ax.set_xticklabels([f"{sigma2:.2f}" for sigma2 in np.linspace(0.1, d, num=num)])
plt.show()


# %% Simulate the SSM with user defined seed

# the strem provide the method
# next() to return the next keys to be used

seed = jax.random.PRNGKey(1234)

stream = KeyStream(seed)
y_sim, x_sim = model.sim(stream)


# user defined stream using jax API
class mystream(KeyStream):
    def __init__(self):
        self._key = jax.random.PRNGKey(1)

    def next(self,):
        new_key, _ = jax.random.split(self._key)
        return new_key


stream = mystream()
y_sim, x_sim = model.sim(stream, Xbeta)

# plot one time-series
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_sim[0, :], label='Simulated observation (y)')
ax.plot(x_sim[0, :], ':', label='Simulated state (x)')
ax.set_title('Simulated Time Series')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
plt.show()

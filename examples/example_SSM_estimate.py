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




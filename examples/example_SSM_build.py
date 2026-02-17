#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 23:41:43 2026

@author: jacopo
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import jax
import pickle

# %% import and check the version
# pip install -e geossm

import geossm

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

# %% print the model summary
print(model)
print(model.summary())

# %% Save the model using pickle
filename = "model.pkl"

with open(filename, 'wb') as file:
    pickle.dump(model, file)

# load the model
with open(filename, 'rb') as file:
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
model.shape

# set model attribute
F = 0.8 * np.eye(q)
model.set(F=F)
print(model.F.diagonal())

# %% Simulate the model

# Use the default Xbeta (see the ssm() for details)
y_sim, x_sim, tdelta = model.sim(seed=1234)
print("Simulate response y:", y_sim.shape)
print("Simulate stete x:", x_sim.shape)
print("Computation time tDelta (s):", tdelta)


# %% Simulate with new Xbeta

Xbeta = np.random.normal(1, 2, size=(p, b, 100))
y_sim, x_sim, tdelta = model.sim(seed=1234, Xbeta=Xbeta)

print("Simulate response y:", y_sim.shape)
print("Simulate stete x:", x_sim.shape)
print("Computation time tDelta (s):", tdelta)

# plot one time-series
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_sim[0, :], label='Simulated observation (y)')
ax.plot(x_sim[0, :], ':', label='Simulated state (x)')
ax.set_title('Simulated Time Series')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
plt.show()





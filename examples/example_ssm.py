#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 23:41:43 2026

@author: jacopo
"""

import geossm.ssm
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


# %% import and check the version
# pip install -e geossm

import geossm

# print("Version: ", geossm.__version__)
print("Load from: ", geossm.__file__)


# %% Create the State Space Model (SSM)
# Type = linear time-invariant SSM

if geossm.__file__:
    from geossm.ssm import StateSpaceModel as ssm


# create the paametrisetim matrix
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
model = ssm(F, H, Q, R, Xbeta=Xbeta, beta=beta,
            x0=None, Sigma0=None, dtype=np.float32)


# %% print the model
print(model)
print(model.summary())

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

# use the internal Xbeta
y_sim, x_t = model.sim(seed=1234)


# provide new Xbeta
Xbeta = np.random.normal(1, 2, size=(p, b, 100))
y_sim, x_t = model.sim(seed=1234, Xbeta=Xbeta)

print("Simulate response y:", y_sim.shape)
print("Simulate stete x:", x_t.shape)

# plot one time-series
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_sim[0, :], label='Simulated observation (y)')
ax.plot(x_t[0, :], ':', label='Simulated state (x)')
ax.set_title('Simulated Time Series')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 23:41:43 2026

@author: jacopo
"""

import numpy as np
import matplotlib.pyplot as plt
import jax
import pickle


# %% import and check the version
# pip install -e geossm

import geossm

# print("Version: ", geossm.__version__)
print("Load from: ", geossm.__file__)


# %% Create the State Space Model (SSM)
# Type = linear time-invariant SSM

if geossm.__file__:
    from geossm.ssm import StateSpaceModel as ssm
    from geossm.utils import KeyStream


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
model = ssm(H, R, F, Q, Xbeta=Xbeta, beta=beta,
            x0=None, Sigma0=None, dtype=np.float32)

# %% save the model using pickale
filename = "model.pkl"

with open(filename, 'wb') as file:
    pickle.dump(model, file)


# load the model
with open(filename, 'rb') as file:
    mymodel = pickle.load(file)
    print(mymodel)


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
y_sim, x_sim = model.sim(seed=1234)


# %% Simualte with new beta

Xbeta = np.random.normal(1, 2, size=(p, b, 100))
y_sim, x_sim = model.sim(seed=1234, Xbeta=Xbeta)

print("Simulate response y:", y_sim.shape)
print("Simulate stete x:", x_sim.shape)

# plot one time-series
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_sim[0, :], label='Simulated observation (y)')
ax.plot(x_sim[0, :], ':', label='Simulated state (x)')
ax.set_title('Simulated Time Series')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
plt.show()


# %% Filter

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

model = ssm(H, R, F, Q, Xbeta=Xbeta, beta=beta,
            x0=None, Sigma0=None, dtype=jax.numpy.float32)

print(model.summary())

# simulate the data
y_sim, x_sim = model.sim(seed=1234)

# filter the state
x_t, P_t, K, x_t_1, P_t_1, invP_t_1, logL, tDelta = model.filter(y_sim)

print("Filtered state x_t:", x_t.shape)
print("Filtered state covariance P_t:", P_t.shape)
print("Kalman gain K:", K.shape)
print("Predicted state x_t_1:", x_t_1.shape)
print("Predicted state covariance P_t_1:", P_t_1.shape)
print("Predicted state invP_t_1:", invP_t_1.shape)
print("Log-likelihood logL:", logL)
print("Computation time tDelta (s):", tDelta)

# plot one time-series
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_sim[0, :], label='Simulated observation (y)')
ax.plot(x_sim[0, :], label='Simulated state (x_sim)')
ax.plot(x_t[0, :], ':', label='Filtered state (x_t)')
ax.set_title('Filtered Time Series')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
plt.show()


# %% Filtering performance under increasing noise
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

    # filter the state
    x_t, P_t, K, x_t_1, P_t_1, invP_t_1, logL, tDelta = model.filter(y_sim)

    # compute the rmse
    rmse.append(np.sqrt(np.mean((x_sim - x_t)**2)))

# plot the rmse
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(rmse, marker='o')
ax.set_title('RMSE vs Measurement Noise Variance')
ax.set_xlabel('Measurement Noise Variance (sigma^2)')
ax.set_ylabel('RMSE')
ax.set_xticks(range(len(np.linspace(0.1, d, num=num))))
ax.set_xticklabels([f"{sigma2:.2f}" for sigma2 in np.linspace(0.1, d, num=num)])
plt.show()


# %% Smooth

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

print(model.summary())

# simulate the data
y_sim, x_sim = model.sim(seed=1234)

# smooth the state
x_sm, P_sm, P_T_1_sm, tDelta = model.smoother(y_sim)

# plot one time-series
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_sim[0, :], label='Simulated observation (y)')
ax.plot(x_sim[0, :], label='Simulated state (x_sim)')
ax.plot(x_sm[0, :], ':', label='Smoothed state (x_t)')
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

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
q = 2
b = 3
T = 50

# markovian matrix
F = 0.85 * np.eye(q)

# mapping matrix
H = np.hstack((np.ones((p, 1)), np.random.binomial(1, 0.5, size=(p, q-1))))

# measueremtent error covaraince matrix
R = 8 * np.eye(p)

# innovetion covariance matrix
Q = 10 * np.eye(q)

# regression design matrix
Xbeta = np.random.normal(0, 1, size=(p, b, T))
beta = np.ones(b)

# %% Build the model
model = ssm(H, R, F, Q, Xbeta=Xbeta, beta=beta,
            x0=None, Sigma0=None, dtype=np.float32)

print(model)

# %% Simulate the model
y_sim, x_sim, tdelta = model.sim(seed=1234)

print("Simulate response y:", y_sim.shape)
print("Simulate stete x:", x_sim.shape)
print("runtime", tdelta)

# plot one time-series
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_sim[0, :], label='Simulated observation (y)')
ax.plot(x_sim[0, :T], ':', label='Simulated state (x)')
ax.set_title('Simulated Time Series')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
plt.show()

# %% Simulate with dirrent beta

# regression design matrix
Xbeta = np.random.normal(1, 2, size=(p, b, 100))
beta = 2 * np.ones(b)

y_sim, x_sim, tdelta = model.sim(seed=1234, Xbeta=Xbeta, beta=beta)

T = model.T
 
print("Simulate response y:", y_sim.shape)
print("Simulate stete x:", x_sim.shape)
print("runtime", tdelta)

# plot one time-series
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_sim[0, :], label='Simulated observation (y)')
ax.plot(x_sim[0, :T], ':', label='Simulated state (x)')
ax.set_title('Simulated Time Series')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
plt.show()



# %% Estimate the model (by filtering the data)

results = model.filter(y_sim)
print(results)

# print the summary of the results
print(results) # = print(results.summary())

# %% Coverage probability of the state 

lower, upper = results.conf_int_state(alpha=0.05, which="filtered")
t = np.arange(T)

plt.figure()
plt.plot(t, x_sim[0, :T], label='Simulated state (x)')
plt.plot(t, results.x_filtered[0, 1:], ':', label='Filtered state (x_filtered)')
plt.fill_between(t, lower[0][1:], upper[0][1:], alpha=0.3, label='95% Confidence Interval')
plt.grid()
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Filtered State with 95% Confidence Interval")
plt.legend()
plt.show()

# compute the coverage probability
inner = (x_sim[0,:T] >= lower[0, 1:]) & (x_sim[0, :T] <= upper[0, 1:])
coverage_probability = np.mean(inner)
print("Coverage Probability of the Filtered State:", coverage_probability)


# %% The residual array are all numpy array 

# Get residuals
res = results.residuals

# Plot the grouped by boxplot (firs 50 days)
plt.figure(figsize=(8, 6))
plt.boxplot(res[:,:50], labels=[f"${i}$" for i in range(res.shape[1])][:50])
plt.title("Boxplot of Residuals by Time")
plt.xlabel("Time")
plt.ylabel("Residuals")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid()
plt.show()

# %% Get the goodness of fit

# get the mse
mse_global= results.mse('global')
mse_space= results.mse('space')
mse_time= results.mse('time')

print("MSE global:", mse_global)
print("MSE space:", mse_space.mean())
print("MSE time:", mse_time.mean())

# plot the mse in time
plt.plot(mse_time,label='MSE time')
plt.xlabel("Time")
plt.ylabel("MSE")
plt.title("Mean Squared Error Over Time")
plt.legend()
plt.grid()
plt.show()

# %% Get the prediction and confidence interval

# get the predicted values
y_hat = results.y_hat

# get the confidence interval
lower, upper = results.conf_int_y(alpha=0.05, prediction=True)


t = np.arange(y_hat.shape[1])

plt.figure()
plt.plot(t, y_hat[0])
plt.plot(t, y_sim[0],'x')
plt.fill_between(t, lower[0], upper[0], alpha=0.3)
plt.grid()
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Prediction with 95% Confidence Interval")
plt.show()

# %% Get the residual diagnostics

stats = results.diagnostics()

# print the diagnostics
print("Jarque-Bera test statistic:", stats['jb'])
print("Jarque-Bera p-value:", stats['jb_pvalue'])
print("Omnibus test statistic:", stats['omni'])
print("Omnibus test p-value:", stats['omni_pvalue'])


# %% Get dictionaly of the resutls

results_dict = results.as_dict()

# %% Get the coverage probability

cov_prov_global = results.coverage_probability(which='global')
cov_prov_space = results.coverage_probability(which='space')
cov_prov_time = results.coverage_probability(which='time')

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
H = np.hstack((np.ones((p, 1)), np.random.binomial(1, 0.5, size=(p, q-1))))
R = 2 * np.eye(p)
Q = 6 * np.eye(q)

Xbeta = np.random.normal(0, 1, size=(p, b, T))
beta = np.ones(b)

model = ssm(H, R, F, Q, Xbeta=Xbeta, beta=beta,
            x0=None, Sigma0=None, dtype=jax.numpy.float32)

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

x_t = results.x_filtered[:,1:]

# plot one time-series
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_sim[0, :], label='Simulated observation (y)')
ax.plot(x_sim[0, :T], label='Simulated state (x_sim)')
ax.plot(x_t[0, :], ':', label='Filtered state (x_t)')
ax.set_title('Filtered Time Series')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
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
H = np.hstack((np.ones((p, 1)), np.random.binomial(1, 0.5, size=(p, q-1))))
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
    y_sim, x_sim, tdelta = model.sim(seed=1234)

    # filter the state
    res = model.filter(y_sim)

    # compute the state rmse
    rmse.append(np.sqrt(np.mean((x_sim[:,:T] - res.x_filtered[:,1:])**2)))

# plot the rmse
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(rmse, marker='o')
ax.set_title('State RMSE vs Measurement Noise Variance')
ax.set_xlabel('Measurement Noise Variance (sigma^2)')
ax.set_ylabel('RMSE')
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
H = np.hstack((np.ones((p, 1)), np.random.binomial(1, 0.5, size=(p, q-1))))
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
    y_sim, x_sim, tdelta = model.sim(seed=1234)

    # filter the state
    res = model.filter(y_sim)

    # compute the state rmse
    cov_prob.append(res.coverage_probability(which='global'))


snr = np.mean(Q.diagonal()) / domain

# plot the rmse
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(cov_prob, marker='o')
ax.set_title('State RMSE vs Measurement Noise Variance')
ax.set_xlabel('Measurement Noise Variance (sigma^2)')
ax.set_ylabel('RMSE')
ax.set_xticks(range(len(np.linspace(1, d, num=num))))
ax.set_xticklabels([f"{lam:.2f}" for lam in snr])
plt.show()








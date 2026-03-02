from src.ssm import StateSpaceModel as ssm
import jax.numpy as jnp

# Define dimensions
p = 2  # Observation dimension
q = 2  # State dimension
T = 5  # Number of time steps
b = 1  # Number of exogenous variables
# Initialize model parameters
H = jnp.eye(p, q)
F = jnp.eye(q)
R = jnp.eye(p)
Q = jnp.eye(q)
x0 = jnp.zeros(q)
Sigma0 = jnp.eye(q)
beta = jnp.ones(b)
# Create synthetic data
y_t = jnp.arange(p * T, dtype=jnp.float32).reshape(T, p)
Xbeta = jnp.ones((T, p, b), dtype=jnp.float32)
# Initialize StateSpaceModel
model = ssm(H, F, R, Q, x0, Sigma0, beta, Xbeta)
print(model)
# Run filter
x_t, P_t, K, x_t_1, P_t_1, invP_t_1, logL = model.filter(y_t)
# Run smoother
x_T, P_T, P_T_1 = model.smoother(x_t, P_t, K, x_t_1, P_t_1, invP_t_1)
# Compute expected values
E_x, E_xx, E_xx_1 = model.computeExpectedValues(x_T, P_T, P_T_1)
# Basic assertions to verify output shapes
assert x_t.shape == (q, T + 1)
assert P_t.shape == (q, q, T + 1)
assert x_T.shape == (q, T + 1)
assert P_T.shape == (q, q, T + 1)
assert E_x.shape == (p, T)
assert E_xx.shape == (p, p, T)
assert E_xx_1.shape == (p, p, T)
print("All tests passed successfully.")

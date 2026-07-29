# Example of Automatic Differentiation (AD) of a loss function 
# with respect to model parameters using JAX.

# %% 
from geossm import Param, ModelParams, FitOptions
import jax.numpy as jnp
from jax import grad, hessian

# %% 

# Dafine de model paramiters using the ModelParams dataclass.
params = ModelParams(
    beta=[1., 2.],
    s2e=3.,
)

# Define a generic loss function that takes ModelParams as input and 
# returns a scalar loss value.
def loss_fn(params):
    return (
        jnp.sum(params.beta.value**2)
        + jnp.sum(params.s2e.value**2)
    )

# Evaluate the loss function at the current parameters.
loss_value = loss_fn(params)
print(f"Loss value: {loss_value}")

# Evaluate the gradient of the loss function with respect to the model parameters.
grads = grad(loss_fn)(params)
print(grads)

# Get the gradients for beta
print(f"Gradient w.r.t beta: {grads.beta.value}")

# %% Comute the hessian 


# Evaluate the Hessian of the loss function with respect to the model parameters.
hess = hessian(loss_fn)(params)
print(hess)

# Get the Hessian for beta 
print(f"Hessian w.r.t beta: {hess.beta.value.s2e.value}")


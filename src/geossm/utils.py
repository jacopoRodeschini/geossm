"""
General utility functions for GEOSSM.
"""

import numpy as np
import datetime
import jax
import platform
import psutil
from functools import wraps


# %% [Utils] Select the device for JAX computations

def _select_device(backend):
    # `backend` may be a backend string ('auto'/'cpu'/'gpu') or an
    # already-resolved jax.Device, in which case it is returned unchanged.
    # This lets a device already picked by one model be passed straight
    # through to another (e.g. LRStateSpaceModel forwarding its backend to
    # an internal StateSpaceModel).
    if isinstance(backend, jax.Device):
        return backend
    backend = (backend or "auto").lower()
    if backend == "cpu":
        return jax.devices("cpu")[0]
    if backend == "gpu":
        # jax.devices("gpu") raises a RuntimeError (rather than returning an
        # empty list) when no GPU platform is registered, so this must be
        # caught to surface the intended, clearer error message.
        try:
            gpus = jax.devices("gpu")
        except RuntimeError:
            gpus = []
        if not gpus:
            raise ValueError("backend='gpu' was requested, but no GPU device is available.")
        return gpus[0]
    if backend == "auto":
        # jax.devices() should itself fall back to CPU when a GPU/TPU plugin
        # is installed but no such hardware is present, but this graceful
        # fallback is jaxlib-version dependent: some builds raise a hard
        # RuntimeError instead. Guard against that so 'auto' never crashes.
        try:
            devices = jax.devices()
        except RuntimeError:
            devices = []
        if not devices:
            devices = jax.devices("cpu")
        return devices[0]
    raise ValueError("backend must be one of {'auto', 'cpu', 'gpu'}")


def _to_backend(backend, *xs):
    # Places the inputs on the requested JAX device before compilation.
    device = _select_device(backend)
    return [jax.device_put(x, device=device) for x in xs]


def _on_device(method):
    """
    Decorator that pins every JAX array created while `method` runs to the
    instance's configured backend device (`self._backend`).

    Many intermediate arrays (basis matrices, precision matrices, scratch
    zeros/ones, ...) are built without ever going through `_to_backend`, so
    without this they silently fall back to JAX's default device - the GPU,
    whenever one is present - even when the model was constructed with
    backend='cpu'. Wrapping a public entry point with this decorator makes
    `jax.default_device` cover the whole call, so uncommitted arrays created
    anywhere in the call stack land on the right device too.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with jax.default_device(self._backend):
            return method(self, *args, **kwargs)
    return wrapper


# %% [Utils] key stream


class KeyStream:
    """
    Stateful helper that hands out fresh JAX PRNG keys on demand.

    Wraps a single ``jax.random.PRNGKey`` and splits it on every call to
    `next`, so callers never have to thread/split keys manually or risk
    reusing the same key twice (e.g. across successive simulation calls
    in :meth:`geossm.ssm.StateSpaceModel.sim`).

    Parameters
    ----------
    seed : jax.random.PRNGKey or int
        Initial key (or integer seed, though JAX generally expects an
        already-constructed ``PRNGKey``) used to derive all subsequent
        keys.
    """

    def __init__(self, seed):
        self._key = seed

    def next(self, num=None):
        """
        Return one or more fresh keys, advancing the internal state.

        Parameters
        ----------
        num : int, optional
            If given, return `num` independent keys (shape
            ``(num, ...)``) instead of a single one; the internal state
            is advanced accordingly so the next call continues from a
            key independent of all of them.

        Returns
        -------
        jax.random.PRNGKey or jax.numpy.ndarray
            A single new key, or an array of `num` keys if `num` was
            given.
        """
        if num is None:
            new_key, self._key = jax.random.split(self._key)
        else:
            keys = jax.random.split(self._key, num=num + 1)
            new_key = keys[:-1]
            self._key = keys[-1]

        return new_key


# %% Compute the block diagonal 3D


def block_diag_3D(*arrs):
    """
    Stack 3D arrays into a single 3D block-diagonal array.

    Generalizes :func:`scipy.linalg.block_diag` to a stack of matrices
    that share a common third (e.g. time) axis: each input array of
    shape ``(n_i, m_i, p)`` becomes one diagonal block of an output of
    shape ``(sum(n_i), sum(m_i), p)``, zero-padded elsewhere, for every
    index along the shared ``p`` axis independently. This is used to
    combine several latent factors' (or formulas'/blocks') per-timestep
    system matrices (e.g. loading/basis matrices) into one block-diagonal
    matrix per timestep.

    Parameters
    ----------
    *arrs : numpy.ndarray
        Two or more 3D arrays of shape ``(n_i, m_i, p)``. The first two
        dimensions may differ between arrays; the third dimension `p`
        must be the same for all of them.

    Returns
    -------
    numpy.ndarray
        Block-diagonal array of shape ``(sum(n_i), sum(m_i), p)`` and
        dtype `float64`.
    """
    # Determine the total shape for the first two dimensions
    total_shape_0 = sum(arr.shape[0] for arr in arrs)
    total_shape_1 = sum(arr.shape[1] for arr in arrs)
    # the last dimension should be the same for all arrays
    total_shape_2 = arrs[0].shape[2]

    # Initialize the block diagonal matrix with zeros
    block_diag_matrix = np.zeros((total_shape_0, total_shape_1, total_shape_2))

    # Current start index for the first two dimensions
    current_index_0 = 0
    current_index_1 = 0

    for arr in arrs:
        shape_0, shape_1, shape_2 = arr.shape
        block_diag_matrix[
            current_index_0 : current_index_0 + shape_0,
            current_index_1 : current_index_1 + shape_1,
            :,
        ] = arr
        current_index_0 += shape_0
        current_index_1 += shape_1

    return block_diag_matrix


# % Write the model information into a file


def write(filename, grid_obs, ssm_model, mode="a"):
    """
    Append a run log (observation grid, model, and hardware info) to a text file.

    Formats a timestamped, human-readable report -- the string
    representation (``__str__``) of each object in `grid_obs`, of
    `ssm_model`, and of :func:`getHardware`'s output -- and writes it to
    `filename`. Intended as a lightweight run log for long-running
    fits, not a machine-readable serialization format.

    Parameters
    ----------
    filename : str or os.PathLike
        Path of the file to write (or append) the report to.
    grid_obs : sequence
        Observation-grid object(s) to include in the report; each is
        rendered via its own ``__str__``.
    ssm_model : object
        The (state-space) model to include in the report, rendered via
        its ``__str__``.
    mode : str, default "a"
        File open mode passed to `open` (e.g. ``"a"`` to append,
        ``"w"`` to overwrite).

    Returns
    -------
    filename : str or os.PathLike
        The `filename` argument, unchanged.
    mode : str
        The `mode` argument, unchanged.
    """
    # Get current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    s = f"Current Time: {current_time} \n"

    s += "OBSERVATION GRID \n"
    s += "".join([gr.__str__() for gr in grid_obs])
    s += "\n"

    s += "SSM REPRESENTATION \n"
    s += ssm_model.__str__()
    s += "\n"

    s += "MODEL PARAMITER ESTIMATE"
    s += "\n"

    s += "HARDWARE \n"
    s += getHardware()

    with open(filename, mode) as f:
        f.write(s)

    return filename, mode


# % Get Hardware infromation


def getHardware():
    """
    Build a human-readable summary of the current machine's hardware.

    Collects CPU, memory, and OS information via :mod:`platform` and
    :mod:`psutil`. Used by `write` to record the hardware a model was
    fit on, and useful on its own for logging/reproducibility.

    Returns
    -------
    str
        Multi-line report with system, CPU (physical/logical core
        counts, current frequency), architecture, and total/available
        memory information.
    """
    # Get CPU info
    cpu_name = platform.processor()
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq().current

    # Get architecture
    architecture = platform.architecture()[0]

    # Get memory info
    virtual_memory = psutil.virtual_memory()
    total_memory = virtual_memory.total / (1024**3)  # Convert to GB
    available_memory = virtual_memory.available / (1024**3)  # Convert to GB

    # Get system info
    system = platform.system()
    release = platform.release()
    version = platform.version()
    machine = platform.machine()
    node = platform.node()

    # Compile all information into a string
    info = (
        f"System: {system} {release} {version}\n"
        f"Node Name: {node}\n"
        f"Machine: {machine}\n"
        f"Architecture: {architecture}\n"
        f"CPU: {cpu_name}\n"
        f"Physical CPUs: {cpu_count}\n"
        f"Logical CPUs: {cpu_count_logical}\n"
        f"Current CPU Frequency: {cpu_freq:.2f} MHz\n"
        f"Total Memory: {total_memory:.2f} GB\n"
        f"Available Memory: {available_memory:.2f} GB\n"
    )
    return info

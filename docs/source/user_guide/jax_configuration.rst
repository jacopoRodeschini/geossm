JAX configuration: devices, memory and precision
================================================

geossm runs its computations with `JAX <https://docs.jax.dev>`_. A few JAX
settings are **process-wide**: they affect every JAX computation in your
Python process, not only geossm's. geossm therefore never changes them
itself; this page lists the ones that matter and how to set them.

Most of them must be set **before JAX starts**, i.e. before the first JAX
array is created (in practice: at the top of your script, before importing
geossm or creating any model). Setting them later has no effect.

Running on the CPU only
-----------------------

``backend="cpu"`` makes a model compute on the CPU, but it does not stop JAX
from starting its GPU backend when a GPU is installed: JAX initialises every
available platform the first time a device is requested. With the default
allocator (next section) that alone reserves most of the GPU memory.

To keep the GPU completely untouched, restrict JAX to the CPU:

.. code-block:: bash

   JAX_PLATFORMS=cpu python my_script.py

or, at the very top of the script:

.. code-block:: python

   import os
   os.environ["JAX_PLATFORMS"] = "cpu"   # before importing jax or geossm

GPU memory preallocation
------------------------

By default JAX reserves 75 % of the GPU memory when its GPU backend starts,
whether or not the computation needs it. On a shared GPU this blocks other
users and processes. To allocate memory on demand instead:

.. code-block:: bash

   XLA_PYTHON_CLIENT_PREALLOCATE=false python my_script.py

or limit the reserved fraction, e.g. to 30 %:

.. code-block:: bash

   XLA_PYTHON_CLIENT_MEM_FRACTION=.30 python my_script.py

The same variables can be set with ``os.environ`` at the top of the script,
as above. See JAX's `GPU memory allocation
<https://docs.jax.dev/en/latest/gpu_memory_allocation.html>`_ page for
details.

Measured on the Agrimonia example of the Quick Start (107 sites, 60 latent
points, 2192 days; float32 on an NVIDIA RTX PRO 6000 with 96 GB; JAX 0.10.1):

.. list-table::
   :header-rows: 1

   * - Setting
     - GPU memory
     - EM iteration
   * - default (any ``backend``, even ``"cpu"``)
     - 73.5 GB reserved when the model is created
     - 0.58 s
   * - ``XLA_PYTHON_CLIENT_PREALLOCATE=false``
     - 1.3 GB after fitting; 9.0 GB peak with standard errors
     - 0.58–0.62 s
   * - ``JAX_PLATFORMS=cpu`` (CPU model)
     - 0 GB
     - 1.9 s (CPU)

Double precision (float64)
--------------------------

Models use float32 by default. ``dtype=jnp.float64`` requires JAX's 64-bit
mode, which is off by default and is a process-wide setting. Enable it once,
at the start of the program:

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)

or set ``JAX_ENABLE_X64=1`` in the environment. Creating a float64 model
without it raises a ``ValueError``. float32 models give the same results
whether 64-bit mode is on or off.

Matrix-multiplication precision
-------------------------------

On GPUs, JAX may run float32 matrix products in reduced precision
(TensorFloat-32) by default. geossm's numerical kernels always request full
precision, so you do not need to set ``JAX_DEFAULT_MATMUL_PRECISION``. Without
it, float32 results on the GPU were not reproducible between runs.

Compilation cache
-----------------

The first fit in a Python process spends most of its time compiling the
kernels (on a GPU, about 35 s of a 41 s six-iteration fit of the Agrimonia
example). Later fits in the same process reuse them. To reuse them across
processes, enable JAX's persistent compilation cache:

.. code-block:: bash

   JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax python my_script.py

With the cache populated, the same fit took 10.7 s. The cache is keyed on
the JAX version, the device and the array shapes, so a different dataset size
or mesh compiles again.

**How it works.** Every compiled program (XLA executable) gets a key, a hash
of: the program itself (which depends on the array shapes and dtypes, the
model's block sizes and any fixed parameter values), the jaxlib version, the
backend and driver version (e.g. the CUDA driver and GPU type), the XLA flags
and the compiler options. Before compiling, JAX looks the key up in the cache
directory; on a hit it loads the executable instead of compiling it. The
cache is off unless a directory is set, with the ``JAX_COMPILATION_CACHE_DIR``
environment variable or, before the first computation,

.. code-block:: python

   jax.config.update("jax_compilation_cache_dir", "/path/to/cache")

**Where it is saved.** In that directory only, one file per compiled program
(named after the jitted function and the key, e.g. ``jit__filter_kernelJAX-
<hash>``). JAX chooses no default location; ``~/.cache/jax`` above is just a
convention.

**What is cached.** Only programs whose compilation took at least
``jax_persistent_cache_min_compile_time_secs`` (1 s by default): geossm's
filter, smoother, M-step and Hessian kernels, but not the many small
operations compiled in a fraction of a second, which is why a cached first
fit still takes about 10 s instead of the 3 s of a later fit in the same
process. Lowering the threshold (e.g. to 0) caches those too, at the cost of
many small files.

**Lifetime and size.** Entries never expire: a file stays until it is
deleted. Entries that no longer match (after upgrading JAX or the GPU driver,
changing XLA flags, or for data of other shapes) are simply never read again.
By default the directory grows without limit
(``jax_compilation_cache_max_size = -1``); setting a size in bytes, e.g.
``jax.config.update("jax_compilation_cache_max_size", 2 * 1024**3)``,
deletes the least recently used entries beyond it. Deleting the directory is
always safe: the programs are compiled again on the next run.

Within a process, every kernel is compiled once per array shape and dtype and
reused by later fits, predictions and standard errors (``compute_cov_params``
included).

Not implemented yet
-------------------

Two changes could save time on a GPU but are not implemented:

* **Keeping the E-step on the device.** Every EM iteration copies the
  smoothed states, their covariances and the EM sufficient statistics to the
  host and back (about 67 MB per iteration on the Agrimonia example, roughly
  5-10 % of an E-step on a GPU; negligible on the CPU).
* **Fewer synchronisations in the M-step.** The M-step waits for each of its
  small kernels to finish (to time them), which prevents the GPU from
  overlapping them.

Summary
-------

A typical script header:

.. code-block:: python

   import os
   # CPU only (leaves the GPU untouched); remove to use a GPU.
   os.environ.setdefault("JAX_PLATFORMS", "cpu")
   # On a GPU: allocate memory on demand instead of reserving 75 %.
   os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
   # Reuse compiled kernels across runs.
   os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jax"))

   import jax
   # Only if you use dtype=jnp.float64.
   jax.config.update("jax_enable_x64", True)

   import geossm

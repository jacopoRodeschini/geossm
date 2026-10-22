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

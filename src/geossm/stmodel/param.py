"""
Parameter containers for :class:`~geossm.stmodel.LRStateSpaceModel`.

`Param` wraps a single named parameter (value, whether it is fixed
during estimation, and its standard error once fit); `ModelParams`
groups the full set of parameters a low-rank state-space model
estimates. Both are registered as JAX pytrees so they can be passed
directly through ``jax.grad``/``jax.jit`` during `fit`. `FitOptions`
configures the EM stopping rule used by `fit`.
"""

from dataclasses import dataclass, replace
import jax
import jax.numpy as jnp
from jax import tree_util

# %% Data class supports

# Sentinel string used in aux_data when bse is None.
# A plain string is hashable and is never confused for a JAX array.
_NONE_BSE = "__bse_is_none__"


def _safe_asarray(value):
    # Some jaxlib builds raise RuntimeError resolving the default device when
    # a GPU/TPU plugin is installed but no matching hardware is present,
    # instead of falling back to CPU on their own. Retry once on CPU.
    try:
        return jnp.atleast_1d(jnp.asarray(value))
    except RuntimeError:
        with jax.default_device(jax.devices("cpu")[0]):
            return jnp.atleast_1d(jnp.asarray(value))


@dataclass
class FitOptions:
    """
    Stopping-rule and logging options for :meth:`~geossm.stmodel.LRStateSpaceModel.fit`.

    Parameters
    ----------
    max_iter : int, default 20
        Maximum number of EM iterations to run.
    tol_relat : float, default 1e-3
        Relative log-likelihood-improvement tolerance below which the EM
        iteration is considered converged and stops early (before
        `max_iter` is reached).
    verbose : bool, default True
        If `True`, log per-iteration progress during `fit`.
    """

    max_iter: int = 20
    tol_relat: float = 1e-3
    verbose: bool = True

    def __str__(self):
        lines = ["FitOptions:"]
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, float):
                value = f"{value:.2e}"
            elif hasattr(value, "__name__"):
                value = value.__name__
            lines.append(f"  {name:10s}: {value}")
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()


@tree_util.register_pytree_node_class
@dataclass
class Param:
    """
    A single named model parameter, with its value, fixed/free status,
    and (once available) standard error.

    Registered as a JAX pytree: when `fixed` is `False`, `value` is the
    single differentiable leaf (so `Param` instances can be passed
    directly through ``jax.grad``); `name`, `fixed` and `bse` are static
    (non-differentiable) auxiliary data. A fixed parameter (or one with
    `value=None`) has no leaves at all, so it is excluded from
    gradient-based updates during :meth:`~geossm.stmodel.LRStateSpaceModel.fit`.

    Parameters
    ----------
    name : str
        Parameter name (e.g. ``"beta"``, ``"s2e"``).
    value : jax.numpy.ndarray or None
        Current parameter value. `None` before an initial value has been
        set.
    fixed : bool, default False
        If `True`, the parameter is held at `value` and excluded from
        estimation in `fit`; if `False`, it is estimated.
    bse : jax.numpy.ndarray or None, default None
        Standard error(s) of `value`, filled in by
        :meth:`~geossm.stmodel.LRStateSpaceResults.compute_cov_params`.
        `None` until then.
    """

    name: str
    value: jnp.ndarray | None
    fixed: bool = False
    bse: jnp.ndarray | None = None

    def __post_init__(self):
        # Only convert plain Python/numpy values — never re-wrap JAX tracers.
        if self.value is not None and not isinstance(self.value, jax.core.Tracer):
            self.value = _safe_asarray(self.value)
        if self.bse is not None and not isinstance(self.bse, jax.core.Tracer):
            self.bse = _safe_asarray(self.bse)

    def set(self, new_value):
        """
        Return a copy of this parameter with `value` replaced.

        No-op (returns `self` unchanged) if the parameter is `fixed`.

        Parameters
        ----------
        new_value : array_like
            Replacement value.

        Returns
        -------
        Param
            A new `Param` with the updated value (or `self`, if fixed).
        """
        if self.fixed:
            return self
        return Param(name=self.name, value=new_value, fixed=self.fixed)

    def freeze(self):
        """Return a copy of this parameter with `fixed=True`."""
        return Param(self.name, self.value, True)

    def unfreeze(self):
        """Return a copy of this parameter with `fixed=False`."""
        return Param(self.name, self.value, False)

    @property
    def shape(self):
        """tuple or None: Shape of `value`, or `None` if `value` is `None`."""
        return self.value.shape if self.value is not None else None

    @property
    def size(self):
        """int: Number of elements in `value` (0 if `value` is `None`)."""
        return self.value.size if self.value is not None else 0

    def __len__(self):
        return self.size

    # ------------------------------------------------------------------
    # JAX pytree registration
    #
    # children  = differentiable leaves  -> only `value` when free & not None
    # aux_data  = static metadata        -> name, fixed, bse (all non-array)
    #
    # bse lives in aux_data (it is computed FROM the Hessian, not through it).
    # We use the string _NONE_BSE instead of Python None because aux_data must
    # survive pytree equality checks without touching any JAX machinery.
    # ------------------------------------------------------------------

    def tree_flatten(self):
        if self.fixed or self.value is None:
            children = ()
            aux_data = (self.name, self.value, self.fixed, self.bse)
        else:
            children = (self.value,)
            bse_aux = _NONE_BSE if self.bse is None else self.bse
            aux_data = (self.name, self.fixed, bse_aux)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        if len(children) == 0:
            name, value, fixed, bse = aux_data
        else:
            name, fixed, bse_aux = aux_data
            (value,) = children
            bse = None if bse_aux == _NONE_BSE else bse_aux
        # Bypass __post_init__ to avoid calling jnp.asarray on a JAX tracer.
        obj = object.__new__(cls)
        obj.name = name
        obj.value = value
        obj.fixed = fixed
        obj.bse = bse
        return obj

    def __repr__(self):
        status = "fixed" if self.fixed else "free"
        return f"Param(name='{self.name}', shape={self.shape}, {status})"


@tree_util.register_pytree_node_class
@dataclass
class ModelParams:
    """
    The full set of parameters estimated by :meth:`~geossm.stmodel.LRStateSpaceModel.fit`.

    Each field is a :class:`Param` (wrapping a value, fixed/free status,
    and standard error); passing a plain array-like instead of a `Param`
    for any field is allowed and is converted to a free (non-fixed)
    `Param` automatically (see `_ensure_param`). Registered as a JAX
    pytree, so a `ModelParams` can be passed directly through
    ``jax.grad``/``jax.jit``.

    Parameters
    ----------
    beta : Param or array_like, optional
        Regression coefficients for the model's exogenous covariates.
    s2e : Param or array_like, optional
        Measurement-error (observation noise) variance(s).
    f : Param or array_like, optional
        AR(1) coefficient(s) of the latent spatial factor(s)' temporal
        evolution.
    A : Param or array_like, optional
        Loading matrix linking the latent spatial factor(s) to the
        observations.
    ks : Param or array_like, optional
        Matérn rescale/range parameter(s) of the latent covariance
        function(s) (one per factor).
    x0 : Param or array_like, optional
        Mean of the initial latent state, as in
        :class:`~geossm.ssm.StateSpaceModel`.
    Sigma0 : Param or array_like, optional
        Covariance of the initial latent state, as in
        :class:`~geossm.ssm.StateSpaceModel`.
    """

    beta: Param | None = None
    s2e: Param | None = None
    f: Param | None = None
    A: Param | None = None
    ks: Param | None = None
    x0: Param | None = None
    Sigma0: Param | None = None

    def __post_init__(self):
        self.beta = self._ensure_param("beta", self.beta)
        self.s2e = self._ensure_param("s2e", self.s2e)
        self.f = self._ensure_param("f", self.f)
        self.A = self._ensure_param("A", self.A)
        self.ks = self._ensure_param("ks", self.ks)
        self.x0 = self._ensure_param("x0", self.x0)
        self.Sigma0 = self._ensure_param("Sigma0", self.Sigma0)

    @staticmethod
    def _ensure_param(name, value):
        if value is None:
            return Param(name=name, value=None, fixed=False)
        if isinstance(value, Param):
            value.name = name
            return value
        return Param(name=name, value=_safe_asarray(value), fixed=False)

    def as_dict(self):
        """dict: Map each field name to its `Param.value` (fixed and free)."""
        return {k: getattr(self, k).value for k in self.__dataclass_fields__}

    def free_params(self):
        """dict: Map each non-fixed field name to its `Param.value`."""
        return {
            k: getattr(self, k).value
            for k in self.__dataclass_fields__
            if not getattr(self, k).fixed
        }

    def __len__(self):
        return sum(getattr(self, k).size for k in self.__dataclass_fields__)

    @property
    def size(self):
        return self.__len__()

    @property
    def shape(self):
        return (self.size,)

    def tree_flatten(self):
        children = (
            self.beta,
            self.s2e,
            self.f,
            self.A,
            self.ks,
            self.x0,
            self.Sigma0,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __str__(self):
        lines = ["ModelParams:"]
        for name in self.__dataclass_fields__:
            param = getattr(self, name)
            value = param.value
            if hasattr(value, "shape"):
                shape = value.shape
                if name == "x0":
                    value = float(value.mean()).__format__(".4g")
                if name == "Sigma0":
                    value = float(jnp.diag(value).mean()).__format__(".4g")
                summary = f"array(shape={shape}, mean={value})"
            else:
                summary = repr(value)
            status = "fixed" if param.fixed else "free"
            lines.append(f"  {name:8s} ({status}) : {summary}")
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def copy(self):
        """Return a deep copy of the ModelParams."""
        def _cp(p):
            return Param(
                p.name,
                p.value.copy() if p.value is not None else None,
                p.fixed,
                p.bse.copy() if p.bse is not None else None,
            )
        return ModelParams(
            beta=_cp(self.beta),
            s2e=_cp(self.s2e),
            f=_cp(self.f),
            A=_cp(self.A),
            ks=_cp(self.ks),
            x0=_cp(self.x0),
            Sigma0=_cp(self.Sigma0),
        )

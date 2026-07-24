from dataclasses import dataclass, replace
import jax.numpy as jnp
from jax import tree_util

# %% Data class supports

@dataclass
class FitOptions:
    max_iter: int = 20
    tol_relat: float = 1e-3
    verbose: bool = True
    dtype: any = jnp.float32

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
    name: str
    value: jnp.ndarray | None
    fixed: bool = False
    bse: jnp.ndarray | None = None  # standard error of the parameter estimate (optional, can be computed later)

    def __post_init__(self):
        if self.value is not None:
            self.value = jnp.atleast_1d(jnp.asarray(self.value))
        if self.bse is not None:
            self.bse = jnp.atleast_1d(jnp.asarray(self.bse))

    def set(self, new_value):
        """Update parameter value if not fixed."""
        if self.fixed:
            return self
        return Param(
            name=self.name,
            value=jnp.asarray(new_value),
            fixed=self.fixed,
        )

    def freeze(self):
        """Return frozen version of parameter."""
        return Param(self.name, self.value, True)

    def unfreeze(self):
        """Return unfrozen version of parameter."""
        return Param(self.name, self.value, False)

    @property
    def shape(self):
        return self.value.shape if self.value is not None else None

    @property
    def size(self):
        return self.value.size if self.value is not None else 0

    def __len__(self):
        return self.size
    
    def tree_flatten(self):

        if self.fixed:
            children = ()
            aux_data = (self.name, self.value, self.fixed, self.bse)
        else:
            children = (self.value,)
            aux_data = (self.name, self.fixed, self.bse)

        return children, aux_data


    @classmethod
    def tree_unflatten(cls, aux_data, children):

        if len(children) == 0:
            name, value, fixed, bse = aux_data
        else:
            name, fixed, bse = aux_data
            (value,) = children

        return cls(name=name,
                value=value,
                fixed=fixed,
                bse=bse)

    def __repr__(self):
        status = "fixed" if self.fixed else "free"
        shape = self.shape
        return f"Param(name='{self.name}', shape={shape}, {status})"

@tree_util.register_pytree_node_class
@dataclass
class ModelParams:
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
            # ensure name consistency
            value.name = name
            return value

        # raw value, wrap into Param
        return Param(name=name, value=jnp.atleast_1d(jnp.asarray(value)), fixed=False)


    def as_dict(self):
        return {k: getattr(self, k).value for k in self.__dataclass_fields__}

    def free_params(self):
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

            # Compact representation for arrays
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
        return ModelParams(
            beta=Param(self.beta.name, self.beta.value.copy() if self.beta.value is not None else None, self.beta.fixed, self.beta.bse.copy() if self.beta.bse is not None else None),
            s2e=Param(self.s2e.name, self.s2e.value.copy() if self.s2e.value is not None else None, self.s2e.fixed, self.s2e.bse.copy() if self.s2e.bse is not None else None),
            f=Param(self.f.name, self.f.value.copy() if self.f.value is not None else None, self.f.fixed, self.f.bse.copy() if self.f.bse is not None else None),
            A=Param(self.A.name, self.A.value.copy() if self.A.value is not None else None, self.A.fixed, self.A.bse.copy() if self.A.bse is not None else None),
            ks=Param(self.ks.name, self.ks.value.copy() if self.ks.value is not None else None, self.ks.fixed, self.ks.bse.copy() if self.ks.bse is not None else None),
            x0=Param(self.x0.name, self.x0.value.copy() if self.x0.value is not None else None, self.x0.fixed, self.x0.bse.copy() if self.x0.bse is not None else None),
            Sigma0=Param(self.Sigma0.name, self.Sigma0.value.copy() if self.Sigma0.value is not None else None, self.Sigma0.fixed, self.Sigma0.bse.copy() if self.Sigma0.bse is not None else None),
        )

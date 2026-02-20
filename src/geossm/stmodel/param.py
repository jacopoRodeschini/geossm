from dataclasses import dataclass, replace
import jax.numpy as jnp

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


@dataclass
class Param:
    name: str
    value: jnp.ndarray | None
    fixed: bool = False

    def __post_init__(self):
        if self.value is not None:
            self.value = jnp.atleast_1d(jnp.asarray(self.value))
    

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

    def __repr__(self):
        status = "fixed" if self.fixed else "free"
        shape = self.shape
        return f"Param(name='{self.name}', shape={shape}, {status})"

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
        self.beta   = self._ensure_param("beta",   self.beta)
        self.s2e    = self._ensure_param("s2e",    self.s2e)
        self.f      = self._ensure_param("f",      self.f)
        self.A      = self._ensure_param("A",      self.A)
        self.ks     = self._ensure_param("ks",     self.ks)
        self.x0     = self._ensure_param("x0",     self.x0)
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
        return (self.size, )
    
    def __str__(self):
        lines = ["ModelParams:"]
        
        for name in self.__dataclass_fields__:
            param = getattr(self, name)
            value = param.value
            

            # Compact representation for arrays
            if hasattr(value, "shape"):
                shape = value.shape

                if name == 'x0':
                    value = float(value.mean()).__format__(".4g")
                if name == 'Sigma0':
                    value = float(jnp.diag(value).mean()).__format__(".4g")
                
                summary = f"array(shape={shape}, mean={value})"
            else:
                summary = repr(value)
            
            status = "fixed" if param.fixed else "free"
            lines.append(f"  {name:8s} ({status}) : {summary}")
        
        return "\n".join(lines)
    
    def __repr__(self):
        return self.__str__()
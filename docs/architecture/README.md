# geossm architecture

Design documentation of the package: what each class is responsible for, how the classes depend on each other, and
the agreed plan for structural changes. These files are outside the Sphinx tree (`docs/source/`) and are not part of
the user documentation.

| File | Content |
|---|---|
| [class_diagram.mmd](class_diagram.mmd) | UML class diagram of the code **as implemented**: public interface (`+`) vs implementation details (`-`), inheritance, composition, aggregation, dependencies, multiplicities |
| [pipeline.mmd](pipeline.mmd) | LRSSM pipeline as implemented: construction → spatial discretisation → EM → results, with open risks |
| [Architecture_decisions.md](Architecture_decisions.md) | Principal design decisions, the approved refactoring plan (P1–P9) with its status, and the decisions taken |

Rules:

* The diagrams show only what exists in the code. Planned classes appear only in `Architecture_decisions.md`,
  marked as proposals.
* Every change that alters the class structure updates the as-implemented diagrams in the same commit.

Render a diagram (Mermaid; GitHub renders `.mmd` files and ```` ```mermaid ```` blocks directly):

```bash
npx -y @mermaid-js/mermaid-cli@11.4.2 -i docs/architecture/class_diagram.mmd -o class_diagram.svg
```

## Class responsibilities

| Class | Module | Responsibility | Visibility |
|---|---|---|---|
| `StateSpaceModel` | `ssm.statespace` | Holds the matrices of a linear-Gaussian state-space model and runs the JIT filter, smoother, prediction and simulation kernels on them | public |
| `StateSpaceResults` | `ssm.statespace_results` | Read-only container of filter/smoother outputs with residual metrics, intervals, coverage and a summary | public |
| `LRStateSpaceModel` | `stmodel.lrssm` | Low-rank spatio-temporal model: builds the design matrices and the spatial system matrices, runs the EM fit, prediction and simulation (several responsibilities; see the plan, P1/P6) | public |
| `LRStateSpaceResults` | `stmodel.stmodel_results` | Results of a fit: parameters, EM history, fitted values, likelihood-based inference, prediction storage and export (see P1/P2) | public |
| `FittedState` | `stmodel.stmodel_results` | Frozen snapshot of the fit-time configuration needed after fitting (covariance objects, formula builders, sites, dtypes); the interface between a fitted model and its results | internal |
| `ModelParams`, `Param` | `stmodel.param` | Typed parameter container and parameter (value, fixed flag, standard error), registered as JAX pytrees | public |
| `FitOptions` | `stmodel.param` | EM stopping and logging options | public |
| `DesignMatricesBuilder` | `data_preparation` | Validates a long-format GeoDataFrame and evaluates a formula into site × time design arrays, for training and prediction | public |
| `DesignMatrices` | `data_preparation` | Design arrays and metadata of one formula | public |
| `spdeAppoxCov` | `covmodel.covmodels` | SPDE (ν = 1) Matérn precision on a finite-element mesh, marginalised to the inner nodes | public |
| `FEMSolver` | `covmodel.covmodels` | P1 finite elements on a mesh: mass and stiffness matrices, inner/outer nodes, basis evaluation, mesh statistics and plots | public (used through `spdeAppoxCov`) |
| `KeyStream`, `_StaticArray`, `_FormulaInfo`, `_NAPassthrough` | various | PRNG key splitting; hashable pytree aux data; parsed formula; patsy NA action | internal |

Kernels (filter, smoother, sufficient statistics, prediction, M-step updates, SPDE marginal precision, `ks` objective)
are module-level JIT functions in `ssm.statespace` and `stmodel.lrssm`; the classes orchestrate them.

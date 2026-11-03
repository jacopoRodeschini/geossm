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
| `LRStateSpaceModel` | `stmodel.lrssm` | Low-rank spatio-temporal model: builds the design matrices and the spatial system matrices, checks which sites the latent field enters (`in_support`), runs the EM fit and simulation | public |
| `LRStateSpaceResults` | `stmodel.stmodel_results` | Results of a fit: parameters, EM history, fitted values (views of the fitted state), likelihood-based inference (Hessian compiled once per shape), prediction (`predict` arrays, `get_prediction` object) and export | public |
| `LRPredictionResults` | `stmodel.prediction_results` | Out-of-sample predictions of a fit: mean, covariance/variance, `in_support`, back-transformation, export to points (`to_geo`) and polygons (`to_poly`) | public |
| `FittedState` | `stmodel._fitted` | Frozen state of a fit: configuration (covariance objects, formula builders, sites, dtypes), estimates, the fitted arrays (`y_obs` as fitted, `y_hat`, smoothed states, sufficient statistics) and `in_support`; the interface between a fitted model and its results | public (via `fitted_state`) |
| `ModelParams`, `Param` | `stmodel.param` | Typed parameter container and parameter (value, fixed flag, standard error), registered as JAX pytrees | public |
| `FitOptions` | `stmodel.param` | EM stopping and logging options, tolerance of the `ks` step (`ks_tol`); validated on construction | public |
| `DesignMatricesBuilder` | `data_preparation` | Validates a long-format GeoDataFrame and evaluates a formula into site × time design arrays, for training and prediction | public |
| `DesignMatrices` | `data_preparation` | Design arrays and metadata of one formula | public |
| `SPDEApproxCov` | `covmodel.covmodels` | SPDE (ν = 1) Matérn precision on a finite-element mesh, marginalised to the inner nodes | public |
| `FEMSolver` | `covmodel.covmodels` | P1 finite elements on a mesh: mass and stiffness matrices, inner/outer nodes, basis evaluation, mesh statistics and plots | public (used through `SPDEApproxCov`) |
| `KeyStream`, `_StaticArray`, `_FormulaInfo`, `_NAPassthrough` | various | PRNG key splitting; hashable pytree aux data; parsed formula; patsy NA action | internal |

Kernels are module-level JIT functions; the classes orchestrate them: filter, smoother, sufficient statistics,
prediction and simulation in `ssm.statespace`; initial values, M-step updates and the `ks` objective in
`stmodel._em`; spatial system matrices, latent basis and support, observed log-likelihood and prediction from a fit in
`stmodel._fitted`; design helpers in `stmodel._design`; export and change of support in `stmodel._output`.

# Contributing to geossm

Thank you for contributing to geossm. This document explains how to report issues, submit code, run tests, and contribute datasets.

## Quick start
- Fork the repository and create a branch for your work.
- Target branch for PRs: `develop` (unless maintainers specify otherwise).
- Follow the PR checklist below before opening a PR.

## Reporting issues
- Search existing issues at: https://github.com/jacopoRodeschini/geossm/issues
- Open a new issue with:
  - A short, descriptive title
  - Expected vs actual behavior
  - Steps to reproduce and minimal repro (code or data) if possible
  - Environment (OS, Python version, package versions)

## Submitting code & open a Pull Request
1. Fork and clone:
    ```bash
    git clone https://github.com/jacopoRodeschini/geossm.git
    cd geossm
    ```
2. Create a branch (use `git switch`):
    ```bash
    git switch -c feature/short-description
    ```
3. Make changes, add tests and documentation.
4. Commit with clear messages and atomic changes:
    - Subject line: short imperative summary (e.g. `Add resampling utility`)
    - Optionally include body explaining rationale and any breaking changes
5. Push and open a PR against `develop`.

### Branch / commit conventions
- Branch names: `feature/`, `fix/`, `chore/`, `docs/`, `data/`
- Keep commits focused and testable. Squash/rebase if requested by reviewers.

## Local development & tests
Recommended Python workflow (adjust if you use conda/poetry):

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

## Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```


## Formatting / linting tools used by the project (run before PR):

- Black (formatter)
- isort (imports)
- Ruff/flake8 (linter)
- Add project-specific commands to CI to ensure consistency.

## PR checklist
- [ ] Target branch is `develop`
- [ ] All tests pass locally and on CI
- [ ] Code covered by tests (unit/integration where appropriate)
- [ ] Code formatted (black/isort) and linted (ruff/flake8)
- [ ] Docstrings and user docs updated if public API changed
- [ ] CHANGELOG entry if applicable
- [ ] Small, focused scope per PR

## Review & merge
- PRs will be reviewed by maintainers. Expect review requests and suggested changes.
- CI must pass before merging.
- Maintainers may rebase/squash or request that contributors squash changes prior to merge.

## Datasets
When contributing datasets include:

- [ ] Short description and why it is useful
- [ ] Source and provenance (URL, citation)
- [ ] License and permission to redistribute
- [ ] Data schema and units
- [ ] A small example/snippet and a script to ingest or validate the dataset
- [ ] Tests or validation checks for expected fields and ranges

Store datasets or pointers to large datasets in data/ or provide download scripts; avoid committing large binary data to the repo.

## Coding guidelines
- Follow PEP 8 for Python
- Use type hints where helpful
- Write clear docstrings (Google or NumPy style preferred)
- Keep public APIs stable; document breaking changes

## Security and sensitive data
- Do not include credentials, private keys, or personal data in commits or issues.
- For security issues, contact maintainers privately before opening a public issue.

## Contact / Maintainers
If you need help or to escalate, open an issue and tag @maintainers 
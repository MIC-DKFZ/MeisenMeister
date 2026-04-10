# AGENTS

## Purpose

MeisenMeister is a breast MRI classification framework with CLI-driven
preprocessing, training, benchmarking, and inference workflows.

This file is guidance for AI coding agents working in this repository. It is
meant to help agents make safe, focused, repo-consistent changes without
duplicating the full user documentation in the README or `docs/`.

## Repo Map

- `meisenmeister/architectures`: model definitions and architecture adapters
- `meisenmeister/plan_and_preprocess`: fingerprinting, planning,
  preprocessing, homogenization, and dataset-preparation steps
- `meisenmeister/training`: training, prediction, benchmarking, and trainer
  implementations
- `meisenmeister/utils`: shared helpers for paths, files, artifacts, and
  training support code
- `tests`: `unittest`-based test suite

CLI entrypoints are defined in `pyproject.toml` and route through
`meisenmeister.cli`.

## Environment and Paths

- Python `>=3.12` is expected.
- Default install command:

```bash
pip install -e .
```

- Many workflows depend on these storage roots being set:
  - `MM_RAW`
  - `MM_PREPROCESSED`
  - `MM_RESULTS`
- Lowercase variants are also accepted:
  - `mm_raw`
  - `mm_preprocessed`
  - `mm_results`

If a task involves dataset discovery, preprocessing outputs, checkpoints, model
artifacts, or prediction outputs, verify the relevant roots are available
before assuming paths will resolve.

## Working Conventions

- Prefer targeted, minimal edits over broad refactors.
- Preserve existing `unittest` style in tests unless the file already uses a
  different style.
- Prefer tests that create their own temporary directories and dummy inputs
  instead of relying on machine-specific files, external datasets, or prebuilt
  artifacts.
- Keep generic utilities and generic tests free of project- or dataset-specific
  assumptions unless the task explicitly targets a specialized workflow.
- Keep public behavior stable unless the task explicitly requires a behavior
  change.
- When changing CLI behavior, path resolution, configuration expectations, or
  user-visible workflows, cross-check `README.md` and relevant files in
  `docs/` for drift.
- Follow the surrounding code style in each file rather than introducing a new
  local convention during unrelated changes.

## Validation Expectations

- For focused changes, run the narrowest relevant target first, for example:

```bash
python -m unittest -q tests.test_training_cases_loading
```

- For broader changes, run the full suite:

```bash
python -m unittest discover -q tests
```

- Full-suite runs may emit warning noise from Matplotlib, sklearn, or PyTorch.
  Warnings alone are not test failures.
- For utility functions, validate both happy-path and failure-path behavior when
  practical.
- Prefer the smallest validation step that gives confidence first, then expand
  to broader checks when the change affects shared code paths.

## Practical Guardrails

- Do not assume external datasets exist locally.
- Do not hardcode task-specific, environment-specific, or dataset-specific
  assumptions into generic code paths.
- Prefer behavior-level assertions over brittle assertions tied to full path
  strings, full exception messages, or incidental formatting details.
- Keep README and docs updates in scope only when behavior, commands, or user
  expectations actually change.
- Avoid broad renames, restructures, or style churn unless they are required to
  complete the task safely.

## Useful Commands

```bash
pip install -e .
python -m unittest -q tests.test_training_cases_loading
python -m unittest discover -q tests
```

## References

- `README.md` for setup, CLI, and storage-root expectations
- `docs/` for deeper pipeline, training, and workflow context

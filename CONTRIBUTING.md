# Contributing

Thank you for contributing to the Inertial Positioning Benchmark.

## Scope

Contributions should fit at least one of these areas:

- canonical data formats and dataset adapters;
- reproducible baselines and evaluation protocols;
- Android data capture or on-device evaluation;
- auditable empirical studies;
- inertial-odometry-specific research workflows.

Method proposals belong in `research/methods/` and must be labeled as proposals until code, configurations, and reproducible results are available.

## Repository conventions

- Commit changes directly to `main`; this repository does not use feature branches.
- Use `README.md` for Chinese/default documentation and `README_EN.md` for English.
- Keep executable benchmark code under `src/`, tests under `tests/`, independently runnable components under `tools/`, and non-executable research material under `research/`.
- Treat the root `.github/workflows/` directory as the only CI and release-workflow location; nested component workflows are not active after consolidation.
- Use repository-relative links for files in this repository.
- Do not claim performance improvements without an auditable result table, fixed data split, configuration, and source commit.
- Preserve dataset, model, and third-party asset provenance and licensing information.

## Validation

Run the benchmark tests:

```bash
python -m pip install -e ".[test]"
pytest -q
```

Run the Android and conversion checks:

```bash
cd tools/inertial-positioning-lab
./gradlew testDebugUnitTest lintDebug assembleDebug
python -m pip install -r requirements-dev.txt
pytest -q
```

The GitHub Actions workflow runs these checks on every push to `main`.

## Documentation

Update the nearest README and any affected file in `docs/` together with code changes. Bilingual documents should remain semantically aligned; a translation does not need to be word-for-word, but status, limitations, paths, and commands must agree.

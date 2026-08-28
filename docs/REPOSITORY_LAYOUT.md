# Repository Layout

This document prevents executable benchmark code, supporting tools, and research notes from being mixed together.

## Top-level ownership

| Path | Responsibility | Admission rule |
| --- | --- | --- |
| `src/inertial_benchmark/` | Stable Python package and public data interfaces | Must have tests and a documented API |
| `tests/` | Core, integration, and consistency tests | Must be deterministic and self-contained |
| `docs/` | Specifications, catalogs, and project-wide policy | Must describe current behavior or label planned work |
| `tools/` | Independently runnable capture, conversion, deployment, or research tools | Each child needs a README and validation path |
| `research/methods/` | Method proposals outside official baselines | Must state evidence status and avoid unsupported claims |
| `research/studies/` | Traceable empirical analyses and assets | Must document setup, provenance, limitations, and interpretation |

## Component boundaries

Shared behavior belongs in `src/inertial_benchmark/`. Dataset-specific conversion scripts and device applications remain under `tools/` until they expose a stable reusable interface.

Every direct child of `tools/` is one coherent component:

- `inertial-positioning-lab/` owns Android capture, archive conversion, and on-device evaluation;
- `paperflow-io/` owns the inertial-odometry-specific workflows preserved from PaperFlow's IO branch.

Generic paper-writing workflows do not belong in this repository.

## From proposal to baseline

A proposal may become an official baseline only after the repository contains:

1. runnable implementation and pinned dependencies;
2. fixed dataset splits and preprocessing;
3. complete training and evaluation configuration;
4. source commit, logs, and auditable result tables;
5. tests or deterministic smoke checks.

Reusable code should then move into a dedicated baseline package rather than remain embedded in a narrative README.

## Naming and documentation

- Directory names use lowercase kebab-case; Python packages use lowercase snake_case.
- `README.md` is the default or Chinese entry; bilingual components use `README_EN.md` for English.
- Repository-relative links are preferred and validated in CI.
- Generated data, checkpoints, build output, and local environments must not be committed.
- A path rename must update workflows, tests, documentation, and examples in the same `main` commit.

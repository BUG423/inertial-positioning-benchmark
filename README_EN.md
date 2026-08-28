<div align="center">

# Inertial Positioning Benchmark

**Unified data interfaces, capture tooling, and reproducible evaluation infrastructure for inertial positioning research**

[中文](README.md) · [English](README_EN.md) · [Data specification](docs/FORMAT.md) · [Repository layout](docs/REPOSITORY_LAYOUT.md)

</div>

## Why this repository exists

Inertial-odometry studies often use incompatible formats, coordinate frames, window definitions, and evaluation protocols. This repository gives executable benchmark code, Android capture tooling, research proposals, empirical studies, and IO-specific research workflows explicit boundaries inside one maintainable project.

The repository currently provides a canonical sequence representation, HDF5 I/O, windowed datasets, and an Android capture pipeline. Public-dataset adapters, standard baselines, and an official leaderboard remain under development.

## Current capabilities

| Component | Status | What it provides |
| --- | --- | --- |
| Benchmark core | **Available** | `CanonicalSequence`, HDF5 I/O, windowing, and tests |
| Android tooling | **Available** | IMU + ARCore VIO capture, `.iplab` export, HDF5 conversion, and on-device inference |
| Coordinate-frame study | **Study** | Global-frame versus body-frame analysis for pedestrian IO |
| PostDiffIO / ModeMoEIO | **Proposal** | Method definitions and experiment plans; no reproducible results yet |
| PaperFlow IO | **Research tool** | Inertial-odometry-only ideation and experiment-design skills |
| Public baselines and leaderboard | **Planned** | Dataset adapters, fixed splits, metrics, and reproducible baselines |

> A method proposal is not a benchmark result. Entry into the official baselines requires auditable code, configurations, data splits, and results.

## Data flow

```text
Device recordings / public datasets
                ↓
CanonicalSequence (timestamps, frames, and units)
                ↓
HDF5 persistence and WindowDataset windowing
                ↓
Baseline training / evaluation / on-device deployment
```

## Repository layout

```text
inertial-positioning-benchmark/
├── src/inertial_benchmark/       # Stable, tested Python core
├── tests/                        # Core, integration, and documentation checks
├── docs/                         # Specifications, surveys, and repository policy
├── tools/
│   ├── inertial-positioning-lab/ # Android capture and on-device evaluation
│   └── paperflow-io/             # Inertial-odometry-only research workflows
└── research/
    ├── methods/                  # Proposals that are not official baselines
    └── studies/                  # Traceable empirical studies and assets
```

See the [repository layout specification](docs/REPOSITORY_LAYOUT.md) for ownership rules and guidance on adding components.

## Quick start

```bash
git clone https://github.com/BUG423/inertial-positioning-benchmark.git
cd inertial-positioning-benchmark
python -m pip install -e ".[test]"
pytest -q
```

```python
from inertial_benchmark import CanonicalSequence, WindowDataset

sequence = CanonicalSequence.from_hdf5("sequence.h5")
dataset = WindowDataset(sequence, window_size=200, stride=10, target="velocity")
sample = dataset[0]
```

The core data layer is framework-agnostic. Convert an Android recording with:

```bash
python tools/inertial-positioning-lab/tools/convert_dataset.py \
  recording.iplab --output datasets/android
```

## Component index

| Entry | Purpose |
| --- | --- |
| [Data format](docs/FORMAT.md) | Sequence schema, coordinate frames, units, and preprocessing |
| [Dataset catalog](docs/DATASETS.md) | Scale, ground truth, licensing, and integration status |
| [Paper survey](docs/PAPERS.md) | Representative methods, datasets, and release status |
| [Tooling](docs/TOOLS.md) | Capture, conversion, and on-device evaluation boundaries |
| [Research](research/README.md) | Method proposals and empirical studies |
| [Developer tools](tools/README.md) | Android tooling and PaperFlow IO |
| [Contributing](CONTRIBUTING.md) | Layout conventions, evidence requirements, and checks |

## Roadmap

- [x] Define the canonical sequence, coordinate, and unit conventions
- [x] Implement HDF5 I/O, window datasets, and core tests
- [x] Integrate Android capture and on-device evaluation tooling
- [x] Consolidate PostDiffIO, ModeMoEIO, the coordinate study, and PaperFlow IO
- [ ] Add the first downloadable or automatically convertible public dataset adapter
- [ ] Integrate at least one end-to-end reproducible baseline
- [ ] Freeze train/validation/test splits and evaluation metrics
- [ ] Publish versioned benchmark results and a leaderboard

## Contributing and license

This repository is maintained only on `main`. Read the [contribution guide](CONTRIBUTING.md) before changing it, and record provenance, licenses, and redistribution constraints for datasets, models, and third-party assets.

A repository-wide open-source license has not yet been selected. Source availability alone does not grant redistribution or commercial-use rights; independently licensed components retain the terms declared in their directories.

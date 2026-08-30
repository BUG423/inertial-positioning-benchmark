<div align="center">

# 🧭 Inertial Positioning Benchmark

### Unified data interfaces, capture tools, and reproducible evaluation infrastructure for inertial positioning research

[![CI](https://github.com/BUG423/inertial-positioning-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/BUG423/inertial-positioning-benchmark/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Android](https://img.shields.io/badge/Android-API%2026%2B-3DDC84?logo=android&logoColor=white)
![Status](https://img.shields.io/badge/status-active-2ea44f)

[🇨🇳 中文](README.md) · **🇬🇧 English**

[Quick Start](#-quick-start) · [Repository Layout](#-repository-layout) · [Research](#-research) · [Roadmap](#-roadmap)

</div>

---

## 🌟 Overview

Inertial-odometry projects often use incompatible data formats, coordinate frames, window definitions, and evaluation protocols. This repository brings the following components into one clearly bounded research platform:

- 📦 **Canonical data interfaces** for timestamps, coordinate frames, units, HDF5 storage, and windowing;
- 📱 **Capture tooling** for phone IMU and ARCore VIO reference trajectories;
- 🧪 **Research material** including method proposals, empirical studies, and figures;
- 🧠 **Research workflows** containing only the inertial-odometry skills retained from PaperFlow;
- ✅ **Automated checks** for the Python core, Android project, conversion tools, and documentation links.

The canonical data layer and Android tooling are available today. Public-dataset adapters, standard baselines, fixed evaluation protocols, and an official leaderboard remain under development.

### Data flow

```text
Device recordings / public datasets
                ↓
CanonicalSequence (timestamps, frames, and units)
                ↓
HDF5 persistence + WindowDataset windowing
                ↓
Model training / evaluation / on-device deployment
```

> [!IMPORTANT]
> PostDiffIO and ModeMoEIO under `research/methods/` are **method proposals**, not reproducible benchmark results. A method enters the official baselines only when its code, configuration, data splits, and results are auditable.

## 🧩 Components

| Component | Status | Description |
| --- | --- | --- |
| Benchmark Core | ✅ Available | `CanonicalSequence`, HDF5 I/O, window datasets, and tests |
| Inertial Positioning Lab | ✅ Available | Android capture, `.iplab` archives, conversion, and on-device inference |
| Pedestrian Coordinate Frames | 📊 Study | Global-frame versus body-frame analysis |
| PostDiffIO | 💡 Proposal | Conditional-diffusion posterior refinement and uncertainty modeling |
| ModeMoEIO | 💡 Proposal | Motion-aware mixture-of-experts routing |
| PaperFlow IO | 🛠️ Research tool | IO ideation, literature validation, feasibility review, and experiment planning |
| Public Baselines & Leaderboard | 🚧 Planned | Dataset adapters, fixed splits, common metrics, and leaderboard |

## 🗂️ Repository layout

```text
inertial-positioning-benchmark/
├── src/inertial_benchmark/       # Core Python data interfaces
├── tests/                        # Core, integration, and documentation checks
├── docs/                         # Specifications, surveys, and project policy
├── tools/
│   ├── inertial-positioning-lab/ # Android capture and on-device evaluation
│   └── paperflow-io/             # Inertial-odometry-only research workflows
├── research/
│   ├── methods/                  # Proposals outside the official baselines
│   └── studies/                  # Traceable empirical studies and assets
├── .github/workflows/            # The single source of CI and releases
└── pyproject.toml                # Python package and dependency configuration
```

| Directory | Entry points |
| --- | --- |
| `docs/` | [Documentation](docs/README.md) · [Format](docs/FORMAT.md) · [Datasets](docs/DATASETS.md) · [Papers](docs/PAPERS.md) |
| `tools/` | [Tools](tools/README.md) · [Android tool](tools/inertial-positioning-lab/) · [PaperFlow IO](tools/paperflow-io/) |
| `research/` | [Research](research/README.md) · [Method proposals](research/methods/) · [Studies](research/studies/) |

See the [repository layout specification](docs/REPOSITORY_LAYOUT.md) for ownership, naming, and component-admission rules.

## 🚀 Quick start

### 1. Clone and install

```bash
git clone https://github.com/BUG423/inertial-positioning-benchmark.git
cd inertial-positioning-benchmark
python -m pip install -e ".[test]"
```

Python 3.9 or later is required.

### 2. Run the tests

```bash
pytest -q
```

This checks the Python interfaces, Android-tool integration, and relative documentation links.

### 3. Load canonical data

```python
from inertial_benchmark import CanonicalSequence, WindowDataset

sequence = CanonicalSequence.from_hdf5("sequence.h5")
dataset = WindowDataset(
    sequence,
    window_size=200,
    stride=10,
    target="velocity",
)
sample = dataset[0]
```

The core data layer is framework-agnostic.

### 4. Convert an Android recording

```bash
python tools/inertial-positioning-lab/tools/convert_dataset.py \
  recording.iplab \
  --output datasets/android
```

See [Inertial Positioning Lab](tools/inertial-positioning-lab/) for format details.

### 5. Build the Android application

```bash
cd tools/inertial-positioning-lab
./gradlew testDebugUnitTest lintDebug assembleDebug
```

The debug APK is written to `app/build/outputs/apk/debug/app-debug.apk`.

## 🔬 Research

- 💡 [PostDiffIO](research/methods/postdiffio/): conditional-diffusion velocity-residual refinement and uncertainty proposal;
- 💡 [ModeMoEIO](research/methods/moe-io/): motion-aware mixture-of-experts proposal;
- 📊 [Pedestrian Coordinate Frames](research/studies/pedestrian-coordinate-frames/): global-frame versus body-frame representations;
- 🧠 [PaperFlow IO](tools/paperflow-io/): IO-only ideation, literature validation, feasibility review, and experiment-design skills.

## 🛣️ Roadmap

- [x] Define the canonical sequence, coordinate-frame, and unit conventions
- [x] Implement HDF5 I/O, window datasets, and core tests
- [x] Integrate Android capture and on-device evaluation tooling
- [x] Consolidate method proposals, the coordinate study, and PaperFlow IO
- [ ] Add the first automatically convertible public-dataset adapter
- [ ] Integrate at least one end-to-end reproducible baseline
- [ ] Freeze dataset splits and common evaluation metrics
- [ ] Publish versioned benchmark results and a leaderboard

## 🤝 Contributing and license

This repository is maintained only on `main`. Read the [contribution guide](CONTRIBUTING.md) before making changes, and record provenance, licenses, and redistribution constraints for datasets, models, and third-party assets.

A repository-wide open-source license has not yet been selected. Source availability alone does not grant redistribution or commercial-use rights; independently licensed components retain the terms declared in their directories.

---

<div align="center">

If this project supports your research, consider giving it a ⭐ and following future benchmark updates.

</div>

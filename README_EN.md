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

- 📦 **IPB v1 data format** with a common 200 Hz time base, gravity-aligned frames, units, HDF5 storage, splits, and conversion reports;
- 🏋️ **Training and evaluation framework** with an Ultralytics-style `ipb` CLI and `NIO` Python API, shared task views, training loop, and trajectory metrics;
- 📱 **Capture tooling** for phone IMU and ARCore VIO reference trajectories;
- 🧪 **Research material** including method proposals, empirical studies, and figures;
- 🧠 **Research workflows** containing only the inertial-odometry skills retained from PaperFlow;
- 📚 **Reproduction guides** preserving the FAST-LIVO dependency, build, and test workflow for Ubuntu 20.04 + ROS Noetic;
- ✅ **Automated checks** for the Python core, Android project, conversion tools, and documentation links.

The IPB v1 data layer, the training/evaluation framework, the RoNIN reference baseline, and the Android tooling are available today. Converters for the public datasets, further baselines, and a leaderboard remain under development. The binding design rules are in the [design charter](docs/DESIGN.md).

### Data flow

```text
Public datasets / device recordings
        ↓  ipb convert (parse → 200 Hz resampling → masks → gravity checks → HDF5 + splits + manifest)
IPB v1 sequences (Sequence)
        ↓  task views (frame, window, target, augmentation)
ipb train / ipb val (model → window velocities → common trajectory reconstruction)
        ↓
trajectory and window metrics → ipb benchmark / ipb report (tables, statistical tests, figures)
```

> [!IMPORTANT]
> PostDiffIO and ModeMoEIO under `research/methods/` are **method proposals**, not reproducible benchmark results. A method enters the official baselines only when its code, configuration, data splits, and results are auditable.

## 🧩 Components

| Component | Status | Description |
| --- | --- | --- |
| IPB v1 data layer | ✅ Available | `Sequence` I/O and validation, 200 Hz resampling, conversion pipeline, grouped splits, and leakage checks |
| Training & evaluation | ✅ Available | `ipb` CLI / `NIO` API, configuration system, Trainer / Validator / Predictor, metrics, and reports |
| RoNIN ResNet baseline | ✅ Available | Independent implementation whose parameter counts match the official code |
| Public dataset converters | 🚧 In progress | RoNIN, RIDI, OxIOD, TLIO, IDOL, RNIN, IMUNet, PedLocData |
| Inertial Positioning Lab | ✅ Available | Android capture, `.iplab` archives, conversion, and on-device inference |
| Pedestrian Coordinate Frames | 📊 Study | Global-frame versus body-frame analysis |
| PostDiffIO | 💡 Proposal | Conditional-diffusion posterior refinement and uncertainty modeling |
| ModeMoEIO | 💡 Proposal | Motion-aware mixture-of-experts routing |
| PaperFlow IO | 🛠️ Research tool | IO ideation, literature validation, feasibility review, and experiment planning |
| FAST-LIVO Setup | 📚 Environment guide | Historical FAST-LIVO reproduction notes for ROS Noetic |
| More baselines & leaderboard | 🚧 Planned | TLIO, RNIN, IMUNet, and versioned results |

## 🗂️ Repository layout

```text
inertial-positioning-benchmark/
├── src/inertial_benchmark/       # IPB core: cfg / data / nn / models / engine / metrics / utils / cli
├── tests/                        # Core, integration, and documentation checks
├── docs/                         # Specifications, surveys, reproduction guides, and policy
│   └── setup/fast-livo.md         # Historical FAST-LIVO environment guide
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
| `docs/` | [Documentation](docs/README.md) · [Design charter](docs/DESIGN.md) · [Metrics](docs/METRICS.md) · [CLI](docs/CLI.md) · [v0.1 format](docs/FORMAT.md) · [Datasets](docs/DATASETS.md) · [Papers](docs/PAPERS.md) · [FAST-LIVO setup](docs/setup/fast-livo.md) |
| `tools/` | [Tools](tools/README.md) · [Android tool](tools/inertial-positioning-lab/) · [PaperFlow IO](tools/paperflow-io/) |
| `research/` | [Research](research/README.md) · [Method proposals](research/methods/) · [Studies](research/studies/) |

See the [repository layout specification](docs/REPOSITORY_LAYOUT.md) for ownership, naming, and component-admission rules.

## 🚀 Quick start

### 1. Clone and install

```bash
git clone https://github.com/BUG423/inertial-positioning-benchmark.git
cd inertial-positioning-benchmark
python -m pip install -e ".[train,test]"   # data layer only: pip install -e .
```

Python 3.9 or newer is required. The data layer depends only on numpy / h5py / scipy / pyyaml; training and reports need the `train` extra (PyTorch, pandas, matplotlib).

### 2. Run the tests

```bash
pytest -q                 # everything, including a ~20 s end-to-end smoke test
pytest -q -m "not slow"   # skip the end-to-end smoke test
```

All tests use synthetic data and cover the format, resampling, conversion, task views, models, analytic metric cases, the engine, the CLI, the Android integration, and documentation links.

### 3. Convert and check a dataset

```bash
export IPB_DATASETS=~/datasets/ipb
ipb convert dataset=ronin source=/path/to/raw/ronin workers=8
ipb check data=ronin full=true
```

### 4. Train, evaluate, and predict

```bash
ipb train model=ronin_resnet18 data=ronin epochs=40 device=0
ipb val   model=runs/train/exp/weights/best.pt data=ronin split=test_unseen
ipb predict model=runs/train/exp/weights/best.pt source=path/to/sequence.h5
```

```python
from inertial_benchmark import NIO, load_sequence

model = NIO("ronin_resnet18")
model.train(data="ronin", epochs=40, device=0)
result = model.val(data="ronin", split="test")      # RunResult
traj = model.predict("path/to/sequence.h5")          # Trajectory
seq = load_sequence("path/to/sequence.h5")           # no PyTorch needed; reads v0.1 files too
```

### 5. Benchmark matrix and reports

```bash
ipb benchmark cfg=main                    # models × datasets × seeds; finished runs are skipped
ipb report runs=runs/benchmark/main out=reports/main
```

See the [CLI reference](docs/CLI.md) and the [metric definitions](docs/METRICS.md).

### 6. Convert an Android recording

```bash
python tools/inertial-positioning-lab/tools/convert_dataset.py \
  recording.iplab \
  --output datasets/android
```

The resulting v0.1 files can be read by `load_sequence()`, which resamples them to v1 in memory. See [Inertial Positioning Lab](tools/inertial-positioning-lab/) for format details.

### 7. Build the Android application

```bash
cd tools/inertial-positioning-lab
./gradlew testDebugUnitTest lintDebug assembleDebug
```

The debug APK is written to `app/build/outputs/apk/debug/app-debug.apk`.

## 🔬 Research

- 💡 [PostDiffIO](research/methods/postdiffio/): conditional-diffusion velocity-residual refinement and uncertainty proposal;
- 💡 [ModeMoEIO](research/methods/moe-io/): motion-aware mixture-of-experts proposal;
- 📊 [Pedestrian Coordinate Frames](research/studies/pedestrian-coordinate-frames/): global-frame versus body-frame representations;
- 🧠 [PaperFlow IO](tools/paperflow-io/): IO-only ideation, literature validation, feasibility review, and experiment-design skills;
- 📚 [FAST-LIVO setup](docs/setup/fast-livo.md): historical reproduction notes for Ubuntu 20.04 and ROS Noetic.

## 🛣️ Roadmap

- [x] Define the canonical sequence, coordinate-frame, and unit conventions
- [x] Implement HDF5 I/O, window datasets, and core tests
- [x] Integrate Android capture and on-device evaluation tooling
- [x] Consolidate method proposals, the coordinate study, and PaperFlow IO
- [x] Archive and integrate the FAST-LIVO ROS 1 reproduction guide
- [x] Write the IPB v1 design charter: 200 Hz format, task views, inference protocol, and metrics
- [x] Implement the conversion pipeline, grouped splits, and leakage checks
- [x] Implement the training/evaluation framework (`ipb` CLI, `NIO` API, benchmark matrix, reports)
- [x] Integrate the RoNIN ResNet reference baseline (parameter counts match the official code)
- [x] Freeze trajectory- and window-level metrics ([METRICS.md](docs/METRICS.md))
- [ ] Add converters for the eight core public datasets and publish splits and fingerprints
- [ ] Integrate TLIO, RNIN, IMUNet, and further reproducible baselines
- [ ] Run the benchmark matrix on real data and publish versioned results and a leaderboard

## 🤝 Contributing and license

This repository is maintained only on `main`. Read the [contribution guide](CONTRIBUTING.md) before making changes, and record provenance, licenses, and redistribution constraints for datasets, models, and third-party assets.

A repository-wide open-source license has not yet been selected. Source availability alone does not grant redistribution or commercial-use rights; independently licensed components retain the terms declared in their directories.

---

<div align="center">

If this project supports your research, consider giving it a ⭐ and following future benchmark updates.

</div>

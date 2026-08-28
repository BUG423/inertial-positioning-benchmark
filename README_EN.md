<div align="center">

# Inertial Positioning Benchmark

A unified, reproducible, and extensible research platform for inertial positioning, covering data conventions, capture tools, evaluation infrastructure, method proposals, and IO research workflows.

[中文](README.md) | [English](README_EN.md)

</div>

## Scope

Inertial positioning research commonly relies on incompatible data formats, coordinate conventions, preprocessing pipelines, evaluation protocols, and baseline implementations. This repository consolidates benchmark code, research notes, and IO-specific tooling into one structured entry point from data capture to method development.

```text
Device recordings and public datasets
    -> Unified formats and coordinates
    -> Loading and windowing
    -> Baselines and research methods
    -> Standardized evaluation
    -> Ideation, experiments, and reproduction
```

## Repository Layout

| Category | Path | Contents |
| --- | --- | --- |
| Benchmark core | `src/`, `tests/` | CanonicalSequence, HDF5 I/O, window datasets, and tests |
| Specifications and surveys | `docs/` | Formats, datasets, paper timeline, and tool documentation |
| Capture tooling | `tools/inertial-positioning-lab/` | Android IMU + ARCore VIO capture, export, conversion, and on-device evaluation |
| Research methods | `research/methods/` | PostDiffIO and MoE-IO proposals |
| Empirical studies | `research/studies/` | Pedestrian IO coordinate-frame experiments and figures |
| IO research workflows | `tools/research-workflows/inertial/` | IO-specific ideation, literature, experiment, and feasibility-review skills |

## Benchmark Core

- [Unified data specification](docs/FORMAT.md): sequence schema, coordinates, units, preprocessing, and initial adapters;
- [Dataset survey](docs/DATASETS.md): scope, ground truth, licenses, downloads, and integration status;
- [Papers and timeline](docs/PAPERS.md): representative methods, experimental datasets, and release status;
- [Tooling and data flow](docs/TOOLS.md): capture, archive, conversion, and evaluation boundaries.

### Inertial Positioning Lab

The bundled [Android tool](tools/inertial-positioning-lab) captures phone IMU and ARCore VIO reference trajectories, exports lossless `.iplab` archives, converts them to canonical HDF5, and runs LiteRT/TFLite models on-device.

> The UI stabilizes only the displayed trajectory. Archives retain the unfiltered ARCore VIO samples. VIO is a reference trajectory, not independent Vicon/RTK-grade ground truth.

## Integrated Research

### Method Proposals

- [PostDiffIO](research/methods/postdiffio): conditional-diffusion posterior refinement of velocity residuals and predictive uncertainty;
- [MoE-IO](research/methods/moe-io): motion-aware mixture-of-experts routing for inertial odometry.

These directories are research proposals and experiment designs. They are not official reproducible benchmark baselines until code, configurations, and complete results are released.

### Empirical Study

- [Pedestrian IO coordinate frames](research/studies/pedestrian-coordinate-frames): global-frame versus device-frame representations, with the original figures and bilingual analysis.

### IO Research Workflows

[IO Research Workflows](tools/research-workflows/inertial) contains only the former PaperFlow `IO` branch:

- `paper-ideation-inertial/`: IO ideation and novelty, impact, and feasibility reviews;
- `paper-experiment-inertial/`: literature re-validation, experiment design, code feasibility, and experiment plans.

Generic paper-writing skills were intentionally excluded to keep this benchmark focused.

## Quick Start

```bash
pip install -e ".[test]"
pytest

python tools/inertial-positioning-lab/tools/convert_dataset.py \
  recording.iplab -o datasets/android
```

```python
from inertial_benchmark import CanonicalSequence, WindowDataset

sequence = CanonicalSequence.from_hdf5("sequence.h5")
dataset = WindowDataset(sequence, window_size=200, stride=10, target="velocity")
sample = dataset[0]
```

The core data layer is framework-agnostic.

## Roadmap

- [x] Define the unified data schema and coordinate conventions (draft v0.1)
- [x] Implement the common data interface and core tests
- [x] Provide Android capture and on-device evaluation tooling
- [x] Consolidate method proposals, coordinate-frame studies, and IO research workflows
- [ ] Add the first public dataset adapter
- [ ] Integrate reproducible representative baselines
- [ ] Freeze metrics, splits, and leaderboard protocols
- [ ] Publish reproducible benchmark results

## Maturity Labels

| Status | Meaning |
| --- | --- |
| Core | Integrated into the common interface and covered by tests |
| Tool | Standalone buildable or runnable supporting tool |
| Study | Preserved experiment, analysis, and resources |
| Proposal | Method concept or experiment design, not yet on the official leaderboard |
| Planned | Designed but not yet implemented |

## Contributing

Contributions of dataset adapters, baselines, coordinate validation, evaluation protocols, and reproduction reports are welcome. Include provenance, licensing, and redistribution constraints for datasets and third-party methods.

## License

A repository-wide open-source license is still being determined; source availability alone does not grant redistribution or commercial-use rights. Integrated content and third-party resources retain the licenses and terms declared in their respective directories.

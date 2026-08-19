<div align="center">

# Inertial Positioning Benchmark

A unified, reproducible, and extensible benchmark for inertial positioning research.

[中文](README.md) | [English](README_EN.md)

</div>

## Motivation

Research in inertial positioning currently relies on heterogeneous datasets, preprocessing pipelines, data formats, evaluation protocols, and baseline implementations. These inconsistencies make fair comparison difficult and create unnecessary overhead when reproducing or extending prior work.

This project aims to provide an open, standardized, and accessible foundation for the inertial positioning community by organizing representative datasets and methods under a unified workflow.

## Goals

- Define a unified data format for inertial positioning datasets.
- Provide reproducible preprocessing pipelines for supported datasets.
- Offer a simple and consistent API for loading processed data.
- Integrate representative open-source baseline methods.
- Establish standardized evaluation protocols and metrics.
- Make it easier to reproduce, compare, and extend existing research.

## Planned Workflow

```text
Raw Dataset
    -> Dataset-specific Preprocessing
    -> Unified Data Format
    -> Common Data Loader
    -> Baseline Methods
    -> Standardized Evaluation
```

Real-device recordings enter the same workflow through the bundled Android tool:

```text
Android IMU + ARCore VIO
    -> Inertial Positioning Lab
    -> Lossless .iplab archive
    -> Canonical HDF5
    -> Common loader and standardized evaluation
```

## Roadmap

- [x] Survey public inertial positioning datasets and open-source methods
- [x] Design the unified data schema (draft v0.1)
- [x] Define preprocessing and coordinate conventions (draft v0.1)
- [x] Implement the common dataset interface (core v0.1)
- [x] Provide Android data capture and on-device model evaluation tooling
- [ ] Add the first supported dataset
- [ ] Integrate representative baselines
- [ ] Define evaluation metrics and benchmark protocols
- [ ] Provide reproducible benchmark results

## Data Interface

- [Unified data specification](docs/FORMAT.md): canonical sequences, coordinates, units, preprocessing rules, and the first six adapters.

## Benchmark Tools

- [Inertial Positioning Lab](tools/inertial-positioning-lab): captures phone IMU and ARCore VIO reference trajectories, exports `.iplab`, converts recordings to canonical HDF5, and runs LiteRT/TFLite models on Android devices;
- [Tooling and data-flow guide](docs/TOOLS.md): role, build and installation instructions, format boundaries, and maintenance workflow;
- [GitHub Releases](https://github.com/BUG423/inertial-positioning-benchmark/releases): downloadable Android preview builds produced by the repository release workflow.

The tool stabilizes only the on-screen trajectory to suppress stationary drift and tracking jumps. Archives retain the unfiltered ARCore VIO samples. VIO is a reference trajectory, not independent Vicon/RTK-grade ground truth.

## Research

- [Papers and timeline](docs/PAPERS.md): papers, methods, experimental datasets, and release status by year.
- [Datasets](docs/DATASETS.md): scope, ground truth, licensing, availability, and integration status.

## Quick Start

The current code assumes that data already follow the canonical HDF5 [data specification](docs/FORMAT.md):

```bash
pip install -e ".[test]"
pytest

# Convert an Android .iplab export to benchmark canonical HDF5
python tools/inertial-positioning-lab/tools/convert_dataset.py recording.iplab -o datasets/android
```

```python
from inertial_benchmark import CanonicalSequence, WindowDataset

sequence = CanonicalSequence.from_hdf5("sequence.h5")
dataset = WindowDataset(sequence, window_size=200, stride=10, target="velocity")

sample = dataset[0]
print(sample.features.shape)  # (6, 200)
print(sample.target.shape)    # (3,)
```

The core layer is framework-agnostic and can be wrapped by a PyTorch data pipeline later.

## Project Status

This project is at an early stage. The data schema, supported tasks, datasets, baselines, and evaluation protocols will be developed incrementally, with design decisions and discussions documented openly.

## Contributing

Contributions and discussions are welcome. We especially welcome participation from dataset authors, method authors, and researchers working on inertial navigation and positioning.

You can contribute by:

- suggesting datasets, methods, tasks, or evaluation protocols;
- helping validate preprocessing pipelines and coordinate conventions;
- contributing dataset adapters or baseline implementations;
- reporting reproducibility issues;
- sharing feedback on the overall benchmark design.

Contribution guidelines will be added as the project structure matures. For now, please open an Issue to introduce your proposal, suggestion, or interest in collaboration.

## License

The repository-wide open-source license is still being determined; source availability alone does not grant redistribution or commercial-use rights. Integrated datasets, baselines, and tools retain the licenses and terms declared in their respective directories.

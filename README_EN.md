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

## Roadmap

- [ ] Survey public inertial positioning datasets and open-source methods
- [ ] Design the unified data schema
- [ ] Define preprocessing and coordinate conventions
- [ ] Implement the common dataset interface
- [ ] Add the first supported dataset
- [ ] Integrate representative baselines
- [ ] Define evaluation metrics and benchmark protocols
- [ ] Provide reproducible benchmark results

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

A license will be selected before the first code release. Individual datasets and baseline methods will retain their original licenses and terms of use.

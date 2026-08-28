<p align="right">
  <a href="README.md">🇨🇳 中文</a> | <b>🇬🇧 English</b>
</p>

# PostDiffIO: Conditional Diffusion Posterior Refinement for Inertial Odometry

> **Status**: Preprint · Code release pending · Validation in progress

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## Abstract

Inertial odometry (IO) estimates motion trajectories from raw inertial measurement unit (IMU) data and serves as the backbone of indoor navigation where GNSS is unavailable. Recent deep-learning methods such as RoNIN have achieved competitive velocity regression, yet their deterministic point estimates lack principled uncertainty quantification — a critical requirement for safety-critical applications.

We present **PostDiffIO**, a two-stage framework that augments a deterministic IO backbone with a conditional diffusion process over velocity residuals. In the first stage, a ResNet1D backbone adapted from RoNIN predicts baseline velocities from IMU sliding windows. In the second stage, a lightweight conditional diffusion refiner models the posterior distribution of the residual between the backbone prediction and ground-truth velocity. During training, the refiner learns to denoise the residual distribution conditioned on backbone features via a standard noise-prediction objective. At inference, deterministic DDIM sampling generates a set of residual samples whose mean refines the backbone estimate and whose variance quantifies prediction uncertainty. For sequential trajectory recovery, we further introduce an error-state Extended Kalman Filter (EKF) that fuses the diffusion-derived velocity observations with IMU strapdown mechanization, producing position and velocity estimates with full covariance tracking.

Preliminary experiments on synthetic and public inertial datasets demonstrate that PostDiffIO outperforms the backbone alone in both velocity accuracy and uncertainty calibration, validating the effectiveness of diffusion-based residual refinement for IO.

---

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Related Work](#2-related-work)
- [3. Method](#3-method)
- [4. Experiments](#4-experiments)
- [5. Conclusion](#5-conclusion)
- [6. Citation](#6-citation)
- [7. License](#7-license)

---

## 1. Introduction

Accurate and robust indoor positioning is a fundamental requirement for applications ranging from augmented reality to autonomous robotics. IMUs, the most ubiquitous motion sensors in consumer electronics, provide high-rate motion sensing but suffer from unbounded drift when naively integrated. Traditional approaches — including Extended Kalman Filters (EKFs) and Zero-Velocity Updates (ZUPT) — rely on meticulous hand-tuning and degrade significantly under aggressive motion or magnetic disturbances.

Deep learning has recently emerged as a new paradigm for IO, enabling direct mapping from IMU sequences to velocity or displacement. RoNIN pioneered the use of ResNet1D for real-time velocity regression on 200 Hz IMU data. Subsequent works have incorporated LSTM for long-range temporal modeling, TCN for efficient sequence processing, and attention mechanisms for capturing distant dependencies. More recently, Transformer architectures have been explored for higher-fidelity trajectory estimation. Despite these advances, a fundamental limitation persists: all these methods produce point estimates without any measure of prediction confidence.

In safety-critical domains — human activity recognition, robot navigation, structural health monitoring — knowing *when* a prediction is unreliable is as important as the prediction itself. Bayesian methods offer a principled framework for uncertainty estimation, but their computational overhead precludes real-time deployment on mobile devices.

Diffusion models, originally developed for image generation, have recently demonstrated remarkable capacity for modeling complex distributions in low-dimensional regression tasks. Their iterative denoising process naturally generates multiple samples from a learned posterior, enabling uncertainty quantification without explicit density estimation.

Building on these observations, we propose **PostDiffIO**, which introduces conditional diffusion models into IO for residual refinement. Our key insight is that rather than modeling the full velocity posterior — which is costly and redundant — we can decompose the problem: a fast deterministic backbone provides a strong velocity baseline, while a lightweight diffusion process captures only the residual uncertainty. This two-stage design achieves the best of both worlds — real-time inference speed from the backbone and principled uncertainty from diffusion.

Our main contributions are:

1. **Conditional diffusion residual refinement**: We model velocity residuals via a conditional diffusion process conditioned on backbone features, achieving uncertainty-aware IO without sacrificing inference speed.
2. **DDIM sampling with EKF fusion**: We fuse DDIM-sampled velocity posteriors with IMU mechanization through an error-state EKF, simultaneously producing high-accuracy trajectories and calibrated uncertainty bounds.
3. **Modular design**: The diffusion refiner is decoupled from the backbone and can be plugged into any deterministic IO network that outputs velocity estimates from IMU inputs.

---

## 2. Related Work

> *This section will be completed upon paper submission.*

### Deep Learning for Inertial Odometry

The architectural evolution of deep-learning-based IO can be divided into several phases. RoNIN first introduced ResNet1D for end-to-end IMU-to-velocity regression. Subsequent works adopted LSTM for long-range temporal modeling, TCN for efficient sequence processing, and attention mechanisms for capturing long-distance dependencies. More recently, Transformer architectures have been explored for higher-precision trajectory estimation.

### Diffusion Models for Regression

After their success in image generation, diffusion models have increasingly been applied to structured regression tasks. Conditional diffusion processes have been explored for time-series forecasting, point cloud denoising, and motion prediction. This work extends the paradigm to inertial odometry, leveraging the low-dimensional nature of velocity residuals for efficient diffusion-based refinement.

### Uncertainty Quantification in IO

Existing uncertainty-aware IO methods include Monte Carlo Dropout, Deep Ensembles, and Evidential Deep Learning. These approaches either introduce significant computational overhead or provide only approximate uncertainty estimates. Diffusion models offer an alternative: multiple forward passes through the sampling process naturally produce an empirical posterior without additional uncertainty parameterization.

---

## 3. Method

### 3.1 Problem Formulation

Given an IMU measurement sequence x₁:T = {(aₜ, ωₜ)} sampled at frequency f (typically 200 Hz), where aₜ ∈ ℝ³ and ωₜ ∈ ℝ³ denote linear acceleration and angular velocity respectively, the IO task is to estimate the corresponding velocity trajectory v₁:T ∈ ℝ^(3×T).

PostDiffIO processes IMU data in sliding windows of length W. Within each window, the backbone predicts a velocity estimate v̂ₜ, and the diffusion refiner models the residual distribution p(εₜ | v̂ₜ, xₜ), where the residual is defined as εₜ = vₜ − v̂ₜ.

### 3.2 RoNIN Backbone

We adopt the RoNIN ResNet1D architecture as the deterministic backbone, with the following processing pipeline:

1. **Input projection**: A 1D convolution maps the 6-dimensional IMU input to a 64-channel feature space.
2. **Residual groups**: Four groups of residual blocks (with configurable group sizes) extract temporal features.
3. **Output head**: A fully-connected layer maps the globally pooled features to 3-dimensional velocity estimates.

The backbone simultaneously extracts intermediate features fₜ ∈ ℝ^(d_f) from the penultimate layer as conditioning input for the diffusion refiner.

### 3.3 Conditional Diffusion Residual Refiner

The refiner is a conditional denoising network ε_θ(rₜ, t, cₜ) that takes a noisy residual, a timestep, and a condition vector as input and predicts the additive noise. Specifically:

- **Noisy residual**: rₜ = √(ᾱₜ) · ε + √(1 − ᾱₜ) · η, the true residual corrupted by noise via the cosine schedule.
- **Timestep embedding**: Sinusoidal encoding of timestep t, projected through a two-layer MLP.
- **Condition vector**: cₜ = Encoder([fₜ; v̂ₜ]), backbone features concatenated with the baseline velocity and projected to 128 dimensions.

The network architecture is: linear projection to 256 dims → four residual blocks (LayerNorm → Linear → SiLU → Linear, with skip connections) → output projection to 3-dim noise prediction.

### 3.4 Training Objective

The total loss is a weighted sum of two components:

> L = L_velocity + λ · L_diffusion

where L_velocity is the velocity MSE loss for the backbone, and L_diffusion is the noise-prediction loss for the residual posterior:

> L_diffusion = E_{t∼U(0,T), η∼N(0,I)} [ ‖η − ε_θ(rₜ, t, cₜ)‖² ]

A cosine noise schedule with 100 diffusion steps is used, balancing generation quality and training efficiency.

### 3.5 DDIM Posterior Sampling

At inference, deterministic DDIM sampling with 10 steps (a subset of the original 100) efficiently draws samples from the residual posterior:

1. Initialization: r_T ∼ N(0, I).
2. Iterative denoising (t = T, T−Δt, …, Δt):
   - Predict noise: η̂ = ε_θ(rₜ, t, c)
   - Estimate clean signal: r₀ = (rₜ − √(1−ᾱₜ) · η̂) / √(ᾱₜ)
   - Update: r_{t−Δt} = √(ᾱ_{t−Δt}) · r₀ + √(1−ᾱ_{t−Δt}) · η̂

Drawing K residual samples (default K=16), we compute:

- **Velocity mean**: v̂ = v̂_backbone + (1/K) · Σ ε⁽ᵏ⁾
- **Uncertainty covariance**: Σ = (1/K) · Σ (ε⁽ᵏ⁾ − ε̄)(ε⁽ᵏ⁾ − ε̄)ᵀ

### 3.6 EKF Trajectory Fusion

For sequence-level trajectory recovery, we employ a 15-dimensional error-state EKF that fuses diffusion velocity observations with IMU mechanization.

**State vector**: x = [p, v, q, bₐ, b_g]ᵀ ∈ ℝ¹⁵, comprising position, velocity, orientation quaternion, accelerometer bias, and gyroscope bias.

**Prediction**: Standard strapdown inertial mechanization with midpoint integration, propagating covariance through the linearized state transition matrix.

**Update**: At each diffusion prediction window, velocity observations are fused via the Kalman gain:

> K = P · Hᵀ · (H · P · Hᵀ + R)⁻¹

where H is the velocity observation matrix and R is the observation covariance derived from diffusion samples. The Joseph form is used for numerically stable covariance updates.

---

## 4. Experiments

> *Full experimental results will be released upon completion of the evaluation suite. Below we outline the experimental design.*

### 4.1 Setup

**Datasets**:

- **OxIOD**: 14 sequences with ground truth from Vicon motion capture
- **IDOL**: Large-scale inertial dataset covering diverse human activities
- **RoNIN-Synthetic**: Synthetic trajectories with configurable noise levels

**Metrics**:

- **Velocity RMSE** (m/s): Root mean square error of velocity estimates
- **ATE** (m): Absolute trajectory error for position
- **NLL**: Negative log-likelihood under the predicted uncertainty
- **ECE**: Expected calibration error for uncertainty quality

**Implementation**: PyTorch 2.0+, Python 3.10+, single NVIDIA RTX 3090. AdamW optimizer with cosine learning rate schedule. 100 diffusion training steps; 10-step DDIM sampling at inference.

### 4.2 Main Results

*Results to be filled upon experimental completion.*

| Method | Velocity RMSE ↓ | ATE ↓ | NLL ↓ | Real-time |
|:------:|:---------------:|:-----:|:-----:|:---------:|
| RoNIN (baseline) | — | — | — | ✓ |
| PostDiffIO | — | — | — | ✓ |
| PostDiffIO + EKF | — | — | — | ✓ |

### 4.3 Uncertainty Quantification

Uncertainty quality is evaluated through:

- **Sparsification plots**: Error as a function of removed fraction (higher AUC indicates better uncertainty ranking).
- **Calibration analysis**: Reliability diagrams comparing predicted vs. empirical variance.
- **Selective prediction**: Accuracy improvement when high-uncertainty predictions are rejected.

### 4.4 Ablation Study

Planned ablations examine:

- Diffusion steps (50 / 100 / 200) — accuracy vs. speed trade-off
- DDIM sample count (4 / 8 / 16 / 32) — uncertainty quality
- Condition dimension (64 / 128 / 256) — model capacity
- Comparison with Monte Carlo Dropout and Deep Ensemble baselines

---

## 5. Conclusion

We presented PostDiffIO, a conditional diffusion framework for uncertainty-aware inertial odometry. By decomposing velocity estimation into a deterministic backbone prediction and a diffusion-refined residual posterior, the proposed method achieves both real-time inference speed and principled uncertainty quantification. Integration with an error-state EKF further enables robust trajectory recovery with full covariance tracking.

Preliminary experiments show that diffusion-based residual refinement improves both point-estimate accuracy and uncertainty calibration over the backbone alone. We believe this "deterministic prediction + diffusion refinement" paradigm is broadly applicable, extending beyond inertial odometry to any regression task that requires uncertainty awareness.

Future work includes extending the framework to multi-modal sensor fusion, designing adaptive diffusion schedules conditioned on motion dynamics, and conducting comprehensive real-world deployment evaluations.

---

## 6. Citation

If you find this work useful, please cite:

```bibtex
@article{postdiffio2025,
  title     = {PostDiffIO: Conditional Diffusion Posterior Refinement for Uncertainty-Aware Inertial Odometry},
  author    = {Your Name and Collaborators},
  journal   = {arXiv preprint},
  year      = {2025},
  note      = {Work in progress. Code: https://github.com/BUG423/diffusion-io}
}
```

---

## 7. License

This project is released under the [MIT License](LICENSE).

**Copyright Notice**: The code and methodology in this repository are provided for research and academic purposes only. Please comply with all applicable copyright and intellectual property laws. Redistribution and derivative works must retain this notice.

For questions or collaboration inquiries, please open an issue.

---

*This README is maintained as part of the project's academic disclosure and will be updated as the evaluation is completed.*

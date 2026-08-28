# ModeMoEIO: Motion-Mode-Conditioned Mixture-of-Experts for Learning-Based Inertial Odometry

[中文](README.md) | **English**

**Author: BUG423**
**Status: Preliminary Working Paper**
**Version: v0.1.0**

> **Preliminary Disclosure**
>
> This repository presents an ongoing research effort. The research concept was developed using the author's self-built tool **PaperFlow** for problem decomposition, method organization, and early experiment planning, with limited preliminary validation conducted. A complete benchmark comparison, ablation experiments, cross-dataset validation, and statistical analysis have not yet been completed; therefore, no final performance conclusions are reported here.
>
> This disclosure aims to establish a traceable original record of the **ModeMoEIO** concept, technical approach, and early implementation ideas. Complete experiments, model parameters, training code, and reproduction materials will be released after the research has been thoroughly validated and formed into a complete paper.

---

## Abstract

Learning-based inertial odometry typically employs a single neural network to directly map fixed-length windows of inertial measurements into velocity, displacement, or relative motion states. However, real pedestrian motion exhibits significant mode heterogeneity: stationary, steady straight-line, turning, and transitional movements induced by starting/stopping, acceleration/deceleration, and posture changes all differ markedly in signal distribution, motion constraints, and regression difficulty. Using a single set of regression parameters to uniformly model all motion states may lead to gradient conflicts between different motion modes, causing the network to learn cross-mode average mappings and thereby weakening its ability to express local motion patterns.

To address this, we propose **ModeMoEIO**, a motion-mode-conditioned mixture-of-experts framework for learning-based inertial odometry. The method first uses a shared temporal encoder to extract unified motion representations from inertial measurement windows, then employs a gating network to estimate soft weights for each sample across different motion modes. Multiple independent velocity experts separately learn local mappings for stationary, straight-line, turning, and transitional motion, with expert predictions continuously fused through gating weights. To promote expert specialization without manual motion mode annotations, we design training-phase motion mode pseudo-labels: samples are partitioned into four basic motion states based on target horizontal velocity and mean yaw angular velocity, with an auxiliary classification loss supervising the gating network. A load-balancing constraint is further introduced to mitigate expert collapse commonly observed during mixture-of-experts training. During inference, no target velocity or manual rules are required; the model autonomously completes expert assignment and velocity prediction based solely on inertial features.

Preliminary validation suggests that motion-mode-conditioned modeling has the potential to improve local velocity regression and complex motion adaptability. This paper is currently published as an early research disclosure; subsequent work will complete systematic benchmarking, ablation studies, expert specialization analysis, and cross-dataset generalization evaluation.

**Keywords:** Learning-based inertial odometry; pedestrian inertial navigation; mixture of experts; motion mode; conditional regression; IMU

---

## 1. Introduction

Inertial measurement units can provide angular velocity and specific force observations at high frequencies without relying on external infrastructure, making them valuable for pedestrian navigation, wearable devices, mobile robots, and satellite-denied environments. However, consumer-grade IMUs exhibit significant noise, bias, and scale errors, and direct integration leads to rapid error accumulation, making pure inertial positioning difficult to maintain long-term stability.

In recent years, learning-based inertial odometry has progressively replaced direct double integration of raw inertial signals by using neural networks to learn motion priors from local IMU sequences. IONet transformed inertial odometry into a sequence learning problem, predicting motion states from local IMU windows through deep networks, laying the foundation for subsequent learning-based inertial navigation. RoNIN further constructed large-scale natural pedestrian motion datasets and systematically studied neural network robustness under complex carrying methods and natural movements. TLIO explicitly modeled prediction uncertainty alongside local displacement regression, tightly coupling learned outputs as relative state observations into extended Kalman filters. These works demonstrated that data-driven methods can extract motion laws from inertial signals that are difficult to explicitly model through traditional integration.

Nevertheless, most existing learning-based inertial odometry methods still process all motion samples through a single backbone network and a single output head. This design implicitly assumes that different motion states can be sufficiently described by the same set of continuous regression functions. However, pedestrian inertial signals differ significantly across states. For instance, stationary states rely more on zero-velocity constraints and sensor noise modeling; steady straight-line motion mainly reflects periodic gait and horizontal velocity relationships; turning motion requires handling the coupling between angular velocity, attitude changes, and velocity direction changes; transitional motion often accompanies non-stationary signals and rapid dynamic changes. Forcing a single regressor to cover all these heterogeneous distributions simultaneously may cause representation competition between different modes.

Mixture-of-experts models provide a natural solution path for this problem. The core idea is to divide the complex input space into several local sub-regions, with different experts learning local mappings and a gating network dynamically combining expert outputs based on input. Compared to a single network, mixture-of-experts can maintain shared feature extraction capability while preserving relatively independent parameter spaces for different motion modes.

Based on this observation, we propose ModeMoEIO. Its core is not simply adding multiple regression heads, but introducing interpretable motion mode priors into the expert routing process: during training, weak supervision labels constructed from velocity and yaw angular velocity guide the gating network to distinguish stationary, straight-line, turning, and transitional motion; during inference, all rule dependencies are removed, with the model autonomously determining expert combinations. The main contributions can be summarized as:

1. Reformulating learning-based inertial odometry as a conditional regression problem with motion mode heterogeneity;
2. Using a shared temporal encoder with multiple independent velocity experts to balance shared representation and mode specialization;
3. Guiding experts to form physically meaningful specialization through training-phase motion mode pseudo-labels;
4. Using soft gating rather than hard switching to continuously describe mode boundaries and complex transitional states;
5. Introducing load-balancing constraints to reduce the risk of long-term expert deactivation or routing collapse.

---

## 2. Related Work

### 2.1 Learning-Based Inertial Odometry

Traditional inertial navigation relies on continuous integration of angular velocity and specific force, but low-cost IMU errors accumulate rapidly over time. Learning-based methods attempt to directly estimate window-level motion quantities through data-driven approaches, thereby reducing drift from long-term integration.

IONet transformed inertial odometry into a sequence learning problem, estimating motion states from local IMU windows through neural networks, laying the foundation for subsequent learning-based inertial navigation. RoNIN further constructed large-scale natural pedestrian motion datasets and proposed neural inertial navigation models suitable for complex carrying methods. In addition to local displacement regression, TLIO explicitly modeled prediction uncertainty and tightly coupled learned outputs as relative state observations into extended Kalman filters. Subsequent work has continued to improve inertial motion prediction from directions including Transformers, context modeling, multi-task learning, and uncertainty estimation.

These methods primarily focus on temporal representation, network structure, or filtering fusion, but still typically use a single predictor to cover all motion states. ModeMoEIO differs in that it directly models heterogeneity within the motion distribution and treats expert specialization as a core structural assumption.

### 2.2 Mixture of Experts and Conditional Computation

Mixture-of-experts models perform weighted combination of multiple local experts through gating functions. Classical mixture-of-experts theory holds that different experts can cover different local regions of the input space, with the gating network learning sample-to-expert assignment relationships. Subsequent research on sparse gating and conditional computation further demonstrated that expert routing can enhance model capacity while controlling computational costs, but also introduces training instability, unequal expert utilization, and routing collapse, typically requiring additional load-balancing mechanisms.

ModeMoEIO adopts soft gating, allowing each sample to be jointly explained by multiple experts. Unlike general MoE primarily used for expanding model capacity, the number of experts in this work is small; the goal is not to build ultra-large-scale networks but to explicitly express local dynamic differences in inertial motion.

### 2.3 Mixture of Experts in Inertial Odometry

Recent work has begun exploring the application of mixture-of-experts in inertial odometry. For example, learning-based inertial odometry research for bicycle localization uses MoE structures to reduce model parameters and computational costs; MosaicIMU uses vehicle-conditioned experts to enhance cross-vehicle generalization. These works demonstrate that expert mechanisms are suitable for handling distribution differences in inertial navigation.

The dimension of difference this work focuses on differs from the above. ModeMoEIO does not primarily route based on vehicle type or computational efficiency, but instead uses **continuously changing motion modes within a single vehicle** as the basis for expert division. Its routing supervision comes from motion state pseudo-labels of stationary, straight-line, turning, and transitional motion, with the goal of enabling experts to learn local velocity mappings with clear motion semantics. Therefore, this work emphasizes conditional regression and expert interpretability at the motion state level.

---

## 3. Method

### 3.1 Problem Definition

Given an inertial measurement window of length $T$:

$$
\mathbf{X}\in\mathbb{R}^{B\times C\times T},
$$

where $B$ denotes batch size, $C$ denotes the number of inertial channels, and $T$ denotes window length. The model objective is to predict the velocity vector corresponding to this window:

$$
\hat{\mathbf{v}}\in\mathbb{R}^{B\times D}.
$$

Standard learning-based inertial odometry typically uses a single function $F$ for mapping:

$$
\hat{\mathbf{v}}=F(\mathbf{X}).
$$

We argue that the true mapping is more appropriately expressed as a combination of multiple motion mode conditional functions:

$$
\hat{\mathbf{v}} = \sum_{k=1}^{K} \pi_k(\mathbf{X})F_k(\mathbf{X}),
$$

where $K$ is the number of experts, $F_k$ denotes the $k$-th motion expert, and $\pi_k$ denotes the expert weight produced by the gating network.

### 3.2 Overall Framework

ModeMoEIO consists of three main components:

1. **Shared temporal encoder**: Extracts unified motion features from raw IMU windows;
2. **Motion mode gating network**: Computes soft assignment weights for different experts based on shared features;
3. **Motion mode expert ensemble**: Separately learns velocity mappings for stationary, straight-line, turning, and transitional motion.

The overall method pipeline is:

**IMU window → Shared temporal representation → Motion mode gating and parallel expert prediction → Soft weight fusion → Velocity output**

During training, motion mode pseudo-labels additionally supervise the gating network; during inference, only inertia-feature-driven autonomous routing is retained.

### 3.3 Shared Temporal Encoding

The shared encoder maps the inertial window to a fixed-dimension feature:

$$
\mathbf{h}=E_{\theta}(\mathbf{X}),
$$

where $E_{\theta}$ denotes a one-dimensional temporal encoding network with parameters $\theta$. The encoder is responsible for extracting basic information that different motion modes commonly depend on, such as local frequency structure, angular velocity changes, short-term dynamic responses, and cross-channel coupling relationships.

Using a shared encoder rather than configuring fully independent backbones for each expert is motivated by two considerations. First, different motion modes still share a large amount of underlying inertial features; completely independent encoding would introduce unnecessary parameter redundancy. Second, shared representations allow the gating and experts to be built in the same feature space, facilitating stable joint optimization.

### 3.4 Motion Mode Definition and Pseudo-Labels

Common inertial odometry datasets typically do not provide explicit motion state labels. To guide experts in forming physically meaningful specialization, this work constructs motion mode pseudo-labels during training based on target horizontal velocity and window-averaged yaw angular velocity.

Define horizontal velocity magnitude:

$$
s = \sqrt{v_x^2+v_y^2},
$$

Define window-averaged yaw angular velocity magnitude:

$$
r = \left| \frac{1}{T} \sum_{t=1}^{T} \omega_z(t) \right|.
$$

This work partitions basic motion states into four categories:

| Mode | Physical Meaning | Criterion |
| --- | --- | --- |
| Stationary | Near-zero horizontal motion | Velocity below stationary threshold |
| Straight-line | Stable translation with weak turning | High velocity and low yaw angular velocity |
| Turning | Stable translation with significant heading change | High velocity and high yaw angular velocity |
| Transition | Starting/stopping, acceleration/deceleration, or mode boundary states | Does not satisfy the above three categories |

Let the motion mode labels generated by the rules be:

$$
y_{\mathrm{mode}}\in\{1,\dots,K\}.
$$

These labels are not used during inference; they serve as training-phase weak supervision signals to help the gating network quickly establish routing structures aligned with actual motion states.

### 3.5 Motion Mode Gating

The gating network produces expert logits based on shared features:

$$
\mathbf{z}=G_{\phi}(\mathbf{h}),
$$

where $G_{\phi}$ is the gating function with parameters $\phi$. Expert weights are then obtained through softmax with a temperature parameter:

$$
\boldsymbol{\pi} = \mathrm{softmax} \left( \frac{\mathbf{z}}{\tau} \right),
$$

where:

$$
\sum_{k=1}^{K}\pi_k=1.
$$

The temperature parameter $\tau$ controls the smoothness of expert assignment. Lower temperatures make routing closer to single-expert selection, while higher temperatures encourage multiple experts to participate jointly. The main reason for adopting soft gating is that there are no strict discrete boundaries between real motion states. For example, during the transition from straight-line to turning, a sample may simultaneously contain stable gait, heading changes, and non-stationary transitions. Continuous weights can avoid prediction discontinuities caused by hard switching.

### 3.6 Motion Mode Experts and Soft Fusion

For each expert $F_{\psi_k}$, the input is the shared feature $\mathbf{h}$:

$$
\hat{\mathbf{v}}_k = F_{\psi_k}(\mathbf{h}), \qquad k=1,\dots,K.
$$

Different experts have the same functional form but independent parameters. The final velocity prediction is:

$$
\hat{\mathbf{v}} = \sum_{k=1}^{K} \pi_k\hat{\mathbf{v}}_k.
$$

During training, gating classification supervision tends to assign different motion states to corresponding experts, while the main velocity regression objective forces each expert to improve prediction accuracy within its high-weight regions. Together, they cause experts to gradually develop local regression capabilities oriented toward different motion modes.

### 3.7 Optimization Objective

The overall training objective consists of velocity regression loss, motion mode gating loss, and load-balancing loss:

$$
\mathcal{L} = \mathcal{L}_{\mathrm{vel}} + \lambda_{\mathrm{mode}}\mathcal{L}_{\mathrm{mode}} + \lambda_{\mathrm{bal}}\mathcal{L}_{\mathrm{bal}}.
$$

#### 3.7.1 Velocity Regression Loss

The velocity regression loss measures the difference between the final fused prediction and the true velocity:

$$
\mathcal{L}_{\mathrm{vel}} = \ell \left( \hat{\mathbf{v}}, \mathbf{v} \right).
$$

Here $\ell$ can adopt mean squared error, smooth $L_1$ loss, or other objectives suitable for velocity regression, depending on the specific training setup.

#### 3.7.2 Motion Mode Gating Loss

The gating network learns training-phase pseudo-labels through cross-entropy:

$$
\mathcal{L}_{\mathrm{mode}} = \mathrm{CE} \left( \mathbf{z}, y_{\mathrm{mode}} \right).
$$

This loss does not require the motion mode rules to completely describe real dynamics, but provides an initial partition with physical interpretability, so that expert specialization does not need to rely entirely on unconstrained spontaneous competition.

#### 3.7.3 Load-Balancing Loss

Let the average gating weights within a training batch be:

$$
\bar{\boldsymbol{\pi}} = \frac{1}{B} \sum_{i=1}^{B} \boldsymbol{\pi}^{(i)}.
$$

Ideally, different experts should receive sufficient samples during training. This work uses a uniform distribution as the base balancing objective:

$$
\mathcal{L}_{\mathrm{bal}} = \left\| \bar{\boldsymbol{\pi}} - \frac{1}{K}\mathbf{1} \right\|_2^2.
$$

This term is primarily used to mitigate the phenomenon of gating concentrating on a few experts during early training. It should be noted that the real motion mode distribution may not be uniform, so the balancing loss serves only as a weak regularization term and should not suppress reasonable expert preferences produced by data.

### 3.8 Inference Process

During inference, only IMU windows are input. The model first generates shared features, then computes gating weights based on features, and fuses all expert outputs:

$$
\mathbf{X} \rightarrow \mathbf{h} \rightarrow \boldsymbol{\pi} \rightarrow \{\hat{\mathbf{v}}_k\}_{k=1}^{K} \rightarrow \hat{\mathbf{v}}.
$$

Target velocity, motion mode rules, and pseudo-labels do not participate in the deployment phase. Therefore, rule supervision serves only as a training-phase inductive bias and does not constitute additional dependencies for actual system operation.

---

## 4. Preliminary Experiments and Analysis

### 4.1 Current Validation Status

Limited-scale internal feasibility testing has been completed. Preliminary observations show that on the basis of maintaining shared temporal encoding capability, introducing motion-mode-conditioned experts allows the model to form non-identical expert responses and shows trends of outperforming a single regression head in some test settings.

Since current validation is incomplete, this paper does not disclose specific numerical results, nor does it claim that the method has stably outperformed existing baselines. Current results only indicate that the following research hypotheses merit further verification:

1. Different motion modes may correspond to different local velocity mappings;
2. Expert specialization based on motion modes may reduce cross-mode averaging in a single model;
3. Training-phase weak supervision can help the gating network form more stable and interpretable assignments;
4. Soft fusion may be more suitable for describing continuous transitional states such as turning, starting/stopping, and acceleration/deceleration.

### 4.2 Subsequent Experiment Design

Complete research will be conducted around the following experiments.

#### Benchmark Comparison

Methods to be compared include:

- Shared encoder with single velocity regression head;
- Multi-head regression network with the same parameter scale;
- Ordinary soft mixture-of-experts without mode supervision;
- Expert model with hard routing;
- ModeMoEIO without load-balancing constraint;
- Complete ModeMoEIO.

#### Evaluation Metrics

Planned metrics include:

- Velocity mean squared error and mean absolute error;
- Trajectory absolute error;
- Fixed-time-window relative trajectory error;
- Grouped prediction error under different motion modes;
- Expert utilization rate and gating entropy;
- Inference time, parameter count, and computational complexity.

#### Ablation Studies

Planned analyses include:

- Number of experts;
- Gating temperature;
- Motion mode thresholds;
- Mode classification loss weight;
- Load-balancing loss weight;
- Soft routing versus hard routing;
- With and without pseudo-label supervision;
- Different motion mode partitioning methods;
- Shared encoder versus independent expert encoders.

#### Expert Specialization Analysis

To verify whether experts truly learn different dynamic patterns, subsequent work will focus on:

- Gating weight distributions corresponding to different motion states;
- Expert switching processes within single trajectories;
- Expert errors in stationary, straight-line, turning, and transitional intervals;
- Diversity of expert outputs;
- Clustering structure of gating feature space;
- Impact of incorrect routing on final trajectories.

### 4.3 Expected Risks

This method still faces several potential issues.

First, motion mode labels constructed based on thresholds may not fully describe real motion dynamics, particularly in complex scenarios such as running, stair climbing, lateral movement, and violent device shaking. Second, experts may only differ at the parameter level without forming stable semantic specialization. Third, simple load-balancing may conflict with real non-uniform motion distributions. Finally, performance improvements may stem from increased parameter count rather than motion mode modeling itself, so they must be verified through parameter-matched baselines and comprehensive ablation studies.

---

## 5. Discussion

The core assumption of ModeMoEIO is that motion states in inertial odometry are not a single homogeneous distribution, but rather consist of multiple sub-distributions with different local patterns. From this perspective, the value of expert mechanisms lies not in simply increasing model capacity, but in providing separable parameter spaces for different dynamic states.

Motion mode pseudo-labels have both advantages and limitations. Their advantage lies in requiring no additional manual annotation and in connecting gating behavior to interpretable motion states; their limitation is that the rules are merely a coarse approximation of the continuous motion space. Future work may further investigate learnable motion prototypes, hierarchical motion states, continuous latent variable routing, or joint training of rule-based and unsupervised routing.

Another important issue is expert granularity. Four basic modes facilitate establishing a clear early verification framework, but this does not mean they represent the optimal partition. Finer-grained experts could cover running, stair climbing, lateral movement, sharp turns, and different carrying methods; however, increasing expert count may also cause insufficient training samples, routing instability, and model complexity. How to balance interpretability, data scale, and prediction performance is a key question for subsequent work.

---

## 6. Conclusion and Future Work

This paper proposes ModeMoEIO, a motion-mode-conditioned mixture-of-experts framework for learning-based inertial odometry. The method extracts unified inertial representations through a shared temporal encoder and uses gating networks for soft assignment among multiple motion experts. During training, motion mode pseudo-labels constructed from horizontal velocity and yaw angular velocity guide stationary, straight-line, turning, and transitional experts to form physically meaningful specialization, while load-balancing constraints reduce expert collapse risk. During inference, the model relies entirely on IMU features for autonomous routing, requiring no rule labels or additional motion state inputs.

The current work is still in its early stages, with preliminary validation supporting continued in-depth research but not yet sufficient for final performance conclusions. Subsequent work will complete systematic benchmark comparisons, cross-dataset experiments, trajectory-level evaluation, expert specialization analysis, and comprehensive ablation studies, and will further investigate more flexible motion mode discovery and routing mechanisms.

---

## 7. Original Disclosure, Research Status, and Material Openness Statement

This repository constitutes the author's first public disclosure record of the **ModeMoEIO** concept, including the problem definition, motion-mode-conditioned expert framework, training-phase motion mode pseudo-supervision, soft expert fusion, and load-balancing design. Repository commit records, version tags, and release records serve to fix the public content and subsequent evolution at the corresponding time points.

This paper is currently a preliminary working draft, not a final manuscript. The author is continuing to improve experiments, theoretical analysis, and paper writing. To protect the integrity of incomplete research and the rights related to unpublished implementation materials, no training code, model parameters, complete experiment configurations, or directly reproducible implementations are provided at this stage. After the research is completed, the paper is formed, and appropriate publication is achieved, the author will reassess the scope of code and experiment material openness.

This repository grants no open-source license. Except where explicitly permitted by applicable law or GitHub platform terms, copying, adaptation, redistribution, commercial use, or creation of derivative implementations based on this content is prohibited without the author's written permission. Academic discussion and citation are not affected, but must clearly attribute the author, project name, repository source, and corresponding version.

(C) 2026 BUG423. All rights reserved.

---

## 8. Citation

```bibtex
@misc{bug423_modemoeio_2026,
  author       = {BUG423},
  title        = {{ModeMoEIO}: Motion-Mode-Conditioned Mixture-of-Experts for Learning-Based Inertial Odometry},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/BUG423/moe-io},
  note         = {Preliminary Working Paper, version 0.1.0}
}
```

---

## References


---

## Acknowledgments

The early research concept organization, method structuring, and preliminary experiment planning of this paper were assisted by the author's self-built tool **PaperFlow**. PaperFlow helps the author translate initial ideas into structured research questions, method proposals, and verification plans. The scientific judgment, method design, experiment responsibility, originality claims, and final expressions in this paper are all the author's own.

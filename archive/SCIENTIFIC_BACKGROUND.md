# Scientific Background: Continual Learning with Closed-form Continuous-time Networks

## Table of Contents
1. [Continual Learning Overview](#continual-learning-overview)
2. [Catastrophic Forgetting](#catastrophic-forgetting)
3. [Continual Learning Strategies](#continual-learning-strategies)
4. [Closed-form Continuous-time Networks (CfC)](#closed-form-continuous-time-networks)
5. [Bayesian Continual Learning](#bayesian-continual-learning)
6. [Industrial Applications: Tennessee Eastman Process](#industrial-applications)
7. [Research Contributions](#research-contributions)

---

## Continual Learning Overview

### Problem Definition

**Continual Learning (CL)**, also known as lifelong learning or incremental learning, addresses the challenge of training machine learning models on sequential tasks without forgetting previously learned knowledge.

**Formal Setting:**
- Model encounters tasks $T_1, T_2, ..., T_n$ sequentially
- Each task $T_i$ has dataset $D_i = \{(x_j, y_j)\}$
- Goal: Maximize performance on all tasks after training on $T_n$
- Constraint: Limited or no access to previous task data $D_1, ..., D_{n-1}$

### Evaluation Scenarios

**Class-Incremental Learning (Class-IL):**
- New classes added incrementally
- Test samples can belong to any seen class
- Most challenging scenario
- Example: MNIST digits 0-1 → 2-3 → 4-5 → 6-7 → 8-9

**Task-Incremental Learning (Task-IL):**
- Task identity provided at test time
- Easier than Class-IL
- Example: Different sensor configurations in industrial monitoring

**Domain-Incremental Learning (Domain-IL):**
- Same classes, different input distributions
- Example: Same fault types under different operating conditions

### Performance Metrics

**Average Accuracy:**
$$\text{ACC} = \frac{1}{T} \sum_{i=1}^{T} a_{T,i}$$
where $a_{T,i}$ is accuracy on task $i$ after training on task $T$.

**Forgetting:**
$$\text{FGT} = \frac{1}{T-1} \sum_{i=1}^{T-1} \max_{j \in \{1,...,T-1\}} (a_{j,i} - a_{T,i})$$
Measures performance degradation on previous tasks.

**Backward Transfer (BWT):**
$$\text{BWT} = \frac{1}{T-1} \sum_{i=1}^{T-1} (a_{T,i} - a_{i,i})$$
Negative BWT indicates catastrophic forgetting.

**Forward Transfer (FWT):**
$$\text{FWT} = \frac{1}{T-1} \sum_{i=2}^{T} (a_{i-1,i} - a_{0,i})$$
Measures knowledge transfer to new tasks.

---

## Catastrophic Forgetting

### Neural Network Plasticity-Stability Dilemma

Neural networks trained with standard backpropagation suffer from **catastrophic forgetting**: when learning new tasks, the weights optimized for previous tasks are overwritten, causing dramatic performance loss.

**Mathematical Perspective:**

For a neural network with parameters $\theta$, standard SGD updates:
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(D_t, \theta_t)$$

This update only considers current task $t$, ignoring:
- Weight importance for previous tasks
- Overlapping representations
- Shared knowledge structures

**Why It Happens:**
1. **Weight Interference:** Same parameters used for multiple tasks
2. **Distributed Representations:** Knowledge encoded non-locally across weights
3. **Non-convex Loss Landscape:** Sharp minima for sequential optimization
4. **Lack of Consolidation:** No mechanism to protect important weights

### Empirical Evidence

**MNIST Sequential Experiment:**
- **Joint Training (upper bound):** 95% accuracy
- **SGD Sequential:** 54% → 19% Class-IL (catastrophic forgetting)
- **Forgetting gap:** ~40-75% performance loss

**Tennessee Eastman Process (TEP):**
- **Joint Training:** 95% fault detection accuracy
- **Incremental Learning:** 67% accuracy
- **Convergence Ratio:** 0.709 (28% forgetting gap)

---

## Continual Learning Strategies

### 1. Replay-Based Methods

Store and replay samples from previous tasks during training on new tasks.

#### Experience Replay (ER)
**Core Idea:** Maintain memory buffer $M$ of past samples, interleave with current task data.

**Algorithm:**
```
For each task t:
  For each minibatch:
    Sample B_new from D_t
    Sample B_old from M
    Compute loss: L = L(B_new) + L(B_old)
    Update θ
  Add samples from D_t to M (reservoir sampling)
```

**Variants:**
- **DER (Dark Experience Replay):** Store model logits, not just samples
  $$\mathcal{L} = \mathcal{L}_{CE}(f(x), y) + \alpha \mathcal{L}_{MSE}(f(x), f_{old}(x))$$

- **DER++:** Add knowledge distillation on current data
  $$\mathcal{L} = \mathcal{L}_{CE} + \alpha \mathcal{L}_{logits} + \beta \mathcal{L}_{features}$$

- **iCaRL:** Combines exemplar storage with knowledge distillation
- **GDumb:** Simple baseline - store samples, retrain from scratch periodically
- **GSS:** Gradient-based sample selection for buffer
- **MER:** Meta-learning for sample selection

**Pros:** Strong performance, conceptually simple  
**Cons:** Memory overhead, privacy concerns, biased sampling

### 2. Regularization-Based Methods

Add regularization terms to preserve important weights for previous tasks.

#### Elastic Weight Consolidation (EWC)

**Core Idea:** Penalize changes to weights important for previous tasks.

**Fisher Information Matrix:**
$$F_i = \mathbb{E}_{x \sim D_i} \left[ \left( \nabla_\theta \log p(y|x, \theta) \right)^2 \right]$$

**EWC Loss:**
$$\mathcal{L}_{EWC} = \mathcal{L}_{current} + \sum_{i=1}^{t-1} \frac{\lambda}{2} F_i (\theta - \theta_i^*)^2$$

**Online EWC:** Maintain running Fisher estimate across tasks.

#### Synaptic Intelligence (SI)

**Core Idea:** Track parameter importance based on gradient trajectory.

**Importance $\omega$:**
$$\omega_k = \sum_{t} \frac{\delta_k^t}{\epsilon + |\Delta \theta_k^t|}$$
where $\delta_k^t$ is accumulated gradient magnitude.

**SI Loss:**
$$\mathcal{L}_{SI} = \mathcal{L}_{current} + c \sum_k \omega_k (\theta_k - \theta_k^*)^2$$

#### Learning without Forgetting (LwF)

**Core Idea:** Preserve output distributions on previous tasks.

**Knowledge Distillation Loss:**
$$\mathcal{L}_{LwF} = \mathcal{L}_{new} + \alpha \sum_{i=1}^{t-1} \mathcal{L}_{KD}(f_i(x), f_i^{old}(x))$$

where $\mathcal{L}_{KD}$ is distillation loss (e.g., KL divergence with temperature $T$).

**Pros:** No memory overhead, no data storage  
**Cons:** Accumulating errors, sensitivity to hyperparameters

### 3. Architecture-Based Methods

Dynamically expand or partition network architecture for new tasks.

#### Progressive Neural Networks (PNN)

**Core Idea:** Add new columns (sub-networks) for each task, with lateral connections.

**Architecture:**
```
Task 1: h₁⁽¹⁾ → h₂⁽¹⁾ → output₁
Task 2: h₁⁽²⁾ ← h₁⁽¹⁾
        ↓
        h₂⁽²⁾ ← h₂⁽¹⁾ → output₂
```

**Forward Pass for task $t$:**
$$h_i^{(t)} = f\left(W_i^{(t)} h_{i-1}^{(t)} + \sum_{j<t} U_i^{(j \to t)} h_{i-1}^{(j)}\right)$$

**Pros:** Zero forgetting, forward transfer  
**Cons:** Linear parameter growth, no Class-IL

#### PackNet, Piggyback, CPG

Other architecture methods that allocate network capacity per task.

**Pros:** No forgetting, clear task separation  
**Cons:** Limited capacity, parameter growth

### 4. Meta-Learning Approaches

Learn how to learn continually through meta-optimization.

#### Meta-Experience Replay (MER)

Combines replay with meta-learning for better generalization.

#### Optimization-Based Meta-Learning

Use MAML-like approaches to find initializations that adapt quickly.

**Pros:** Fast adaptation, good forward transfer  
**Cons:** Computational overhead, complex optimization

---

## Closed-form Continuous-time Networks

### Neural Circuit Policies (NCP)

**Motivation:** Biological neurons exhibit continuous-time dynamics and sparse wiring.

**Liquid Time-Constant (LTC) Networks:**

Standard RNN:
$$h_t = f(W_{hh} h_{t-1} + W_{xh} x_t)$$

**Continuous-time RNN (CT-RNN):**
$$\tau \frac{dh(t)}{dt} = -h(t) + f(W_{hh} h(t) + W_{xh} x(t))$$

**LTC with adaptive time constants:**
$$\tau_i(t, h) \frac{dh_i(t)}{dt} = -h_i(t) + \sum_j w_{ij} \sigma_j(h_j(t))$$

where $\tau_i$ adapts based on input and hidden state.

### Closed-form Continuous-time (CfC) Networks

**Key Innovation:** Analytical solution to LTC dynamics for efficient training.

**CfC Cell Equation:**
$$h(t + \Delta t) = \text{CfC}(h(t), x(t), \Delta t; \theta)$$

Solved using **mixed solver** combining:
1. **Closed-form solution** for linear components
2. **Numerical ODE solver** for non-linear activation

**Computational Advantage:**
- Standard RNN: $O(n)$ sequential steps
- CfC: $O(1)$ per timestep (closed-form update)
- Backprop through continuous time (adjoint method)

### Auto-NCP: Sparse Wiring

**Wiring Architecture:**

Unlike fully-connected RNNs, NCP uses structured sparse connectivity:

```
Input → Sensory → Inter → Command → Motor → Output
         (dense)  (sparse) (sparse)  (dense)
```

**Benefits:**
1. **Reduced Parameters:** 23K vs 59K for MNIST (61% reduction)
2. **Interpretability:** Neuron roles defined by wiring
3. **Robustness:** Less prone to overfitting
4. **Biological Plausibility:** Mimics nervous system structure

**AutoNCP Configuration:**
- Sensory neurons: Input processing (dense connections)
- Inter neurons: Intermediate processing (sparse)
- Command neurons: Decision making (sparse)
- Motor neurons: Output generation (dense)

### CfC for Continual Learning

**Advantages over Standard RNNs/LSTMs:**

1. **Better Temporal Dynamics:**
   - Adaptive time constants preserve multi-scale patterns
   - Continuous-time formulation handles variable-length sequences
   - No vanishing gradients through closed-form solutions

2. **Sparse Representations:**
   - Structured sparsity reduces parameter interference
   - Modular neuron types enable selective consolidation
   - Lower capacity → less catastrophic forgetting

3. **Interpretability:**
   - Wiring structure reveals what network learns
   - Neuron activations have semantic meaning
   - Easier to identify task-critical pathways

**Our Implementation:**

- **MNISTcfc:** Sequential MNIST with AutoNCP (28×28 sequence input)
- **CNN-CfC:** ResNet18 features → CfC temporal processing
- **TEPcfc:** Industrial process monitoring with 52-dim time series

---

## Bayesian Continual Learning

### Motivation

Bayesian framework naturally handles uncertainty and weight consolidation:
- **Posterior:** Encode knowledge from previous tasks
- **Predictive Uncertainty:** Identify task boundaries
- **Principled Regularization:** KL divergence to previous posterior

### Variational Continual Learning (VCL)

**Bayesian Neural Network:**

Instead of point estimate $\theta$, maintain distribution $q(\theta)$.

**After task $t$, posterior:**
$$p(\theta | D_1, ..., D_t) \propto p(D_t | \theta) p(\theta | D_1, ..., D_{t-1})$$

**Variational Inference:**

Use variational distribution $q_t(\theta)$ to approximate posterior.

**VCL Objective for task $t$:**
$$\mathcal{L}_{VCL} = \mathbb{E}_{q_t(\theta)} [-\log p(D_t | \theta)] + \text{KL}(q_t(\theta) || q_{t-1}(\theta))$$

**Implementation:**
- Mean-field approximation: $q(\theta) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$
- Learn $\mu$ and $\sigma$ for each weight
- Previous posterior $q_{t-1}$ acts as regularizer

**Advantages:**
- Natural catastrophic forgetting prevention
- Uncertainty quantification
- Task-agnostic (no task boundaries needed in inference)

### Laplace Approximation for CL

**Simpler Alternative:** Gaussian approximation around mode.

**After task $t$:**
$$q_t(\theta) = \mathcal{N}(\theta_t^*, H_t^{-1})$$

where $\theta_t^*$ is MAP estimate, $H_t$ is Hessian.

**Diagonal Approximation (Laplace-CfC):**
$$H_t \approx \text{diag}(F_t)$$
where $F_t$ is Fisher information (like EWC).

**Benefits:**
- Simpler than full VCL
- Direct connection to EWC
- Uncertainty estimates for free

### Online Bayesian Continual Learning

**Recursive Bayesian Updates:**

No explicit task boundaries - update beliefs continuously.

**Streaming Variational Inference:**
$$q_{t+1}(\theta) = \arg\min_q \text{KL}(q || p(\theta | D_{1:t+1}))$$

**Confidence-Weighted Learning:**

Adjust learning rate based on uncertainty:
$$\eta_i = \frac{1}{\sigma_i^2}$$

Less certain weights update faster.

**Applications:**
- Real-time fault detection
- Non-stationary environments
- Continuous adaptation without retraining

---

## Industrial Applications

### Tennessee Eastman Process (TEP)

**Background:**

TEP is a standard benchmark for fault detection and diagnosis in chemical process control.

**Process Description:**
- 5 major unit operations (reactor, condenser, separator, stripper, compressor)
- 12 manipulation variables
- 41 measured variables (22 continuous, 19 composition)
- 52 total process variables for monitoring

**Fault Scenarios:**

21 programmed faults + 1 normal operation:
1. A/C feed ratio, B composition constant
2. B composition, A/C ratio constant
3. D feed temperature
4. Reactor cooling water inlet temperature
5. Condenser cooling water inlet temperature
6. A feed loss
7. C header pressure loss - reduced availability
8. A, B, C feed composition
9. D feed temperature
10. C feed temperature
11. Reactor cooling water inlet temperature
12. Condenser cooling water inlet temperature
13. Reaction kinetics
14. Reactor cooling water valve
15. Condenser cooling water valve
16-20. Unknown faults
21. Valve position constant (optimize cost)

### Continual Learning for TEP

**Problem Formulation:**

**Incremental Task Setup:**
- Task 1: Normal operation (d00.dat)
- Task 2: Fault 1 detection (d01.dat)
- Task 3: Fault 2 detection (d02.dat)
- ...
- Task 22: Fault 21 detection (d21.dat)

**Challenges:**
1. **Data Imbalance:** Some faults rare, others frequent
2. **Temporal Dependencies:** Process dynamics span multiple timescales
3. **High Dimensionality:** 52-variable time series
4. **Online Deployment:** Cannot retrain on all data

**Our Results:**

| Approach | Accuracy | Forgetting Gap |
|----------|----------|----------------|
| Joint Training | 95% | 0% (upper bound) |
| Incremental SGD | 67% | 28% |
| Target (with CL) | 85-90% | <10% |

**Why CfC for TEP?**

1. **Multi-scale Dynamics:**
   - Fast: Flow rate changes (seconds)
   - Medium: Temperature variations (minutes)
   - Slow: Composition drift (hours)
   - CfC adaptive time constants capture all scales

2. **Interpretability:**
   - Safety-critical application
   - Need explainable decisions
   - AutoNCP wiring shows reasoning pathway

3. **Uncertainty:**
   - Bayesian CfC quantifies confidence
   - Alert operators when uncertain
   - Detect novel faults (out-of-distribution)

### Other Industrial CL Applications

**Predictive Maintenance:**
- Learn new failure modes without forgetting existing ones
- Adapt to equipment degradation over time

**Quality Control:**
- Incrementally learn defect types
- Adapt to product variations

**Smart Manufacturing:**
- Learn new production processes
- Transfer knowledge across similar tasks

**Energy Systems:**
- Adapt to changing consumption patterns
- Learn new renewable energy dynamics

---

## Research Contributions

### This Project's Innovations

**1. CfC Integration with Mammoth Framework**
- First integration of continuous-time neural networks in CL benchmark
- Custom backbones: MNISTcfc, CNN-CfC, TEPcfc
- Systematic comparison with LSTM baselines

**2. Industrial Process Monitoring**
- TEP dataset adaptation for continual learning
- Incremental vs joint training methodology
- Real-world fault detection scenario

**3. Bayesian CfC Methods (Planned)**

**Laplace-CfC:**
```python
# Diagonal Hessian approximation
for task in tasks:
    train(task)
    fisher = compute_fisher_diagonal()
    consolidate(theta_star, fisher)
```

**VCL-CfC:**
```python
# Variational inference with CfC
q_prev = previous_posterior
q_curr = optimize_ELBO(q_prev, current_task)
posteriors.append(q_curr)
```

**Online Bayesian CfC:**
```python
# Recursive Bayesian updates
for sample in data_stream:
    uncertainty = q_theta.variance
    lr = adaptive_learning_rate(uncertainty)
    q_theta = bayesian_update(sample, q_theta, lr)
```

**4. Comprehensive Benchmarking**
- Parallel GPU execution (6 experiments × 2 GPUs)
- 21 CL methods × 3 datasets
- Systematic ablation studies

### Open Research Questions

**1. Optimal Architecture for CL:**
- How does sparse wiring affect continual learning?
- Can we dynamically grow NCP structure per task?
- What is the right balance of sensory/inter/command neurons?

**2. Temporal Credit Assignment:**
- How to consolidate continuous-time representations?
- Can adaptive time constants help mitigate forgetting?
- Multi-scale consolidation strategies?

**3. Uncertainty-Driven Learning:**
- Use predictive uncertainty to detect task boundaries
- Confidence-weighted replay buffer sampling
- Active learning for difficult task transitions

**4. Transfer Learning:**
- Do CfC networks transfer better than RNNs?
- Can temporal inductive biases improve forward transfer?
- Cross-domain continual learning with industrial data

**5. Explainable CL:**
- Visualize neuron activations across tasks
- Identify which neurons are task-specific vs shared
- Wiring analysis for knowledge organization

---

## References

### Continual Learning Foundations

**Surveys:**
- De Lange et al. (2021). "A Continual Learning Survey: Defying Forgetting in Classification Tasks." *IEEE TPAMI*.
- Parisi et al. (2019). "Continual Lifelong Learning with Neural Networks: A Review." *Neural Networks*.

**Seminal Papers:**
- McCloskey & Cohen (1989). "Catastrophic Interference in Connectionist Networks." *Psychology of Learning and Motivation*.
- Kirkpatrick et al. (2017). "Overcoming Catastrophic Forgetting in Neural Networks." *PNAS* (EWC).
- Rebuffi et al. (2017). "iCaRL: Incremental Classifier and Representation Learning." *CVPR*.

### Mammoth Framework

- Buzzega et al. (2020). "Dark Experience for General Continual Learning: A Strong, Simple Baseline." *NeurIPS* (DER).
- Boschini et al. (2022). "Class-Incremental Continual Learning into the eXtended DER-verse." *TPAMI* (X-DER).

### CfC/NCP Networks

- Hasani et al. (2021). "Liquid Time-constant Networks." *AAAI*.
- Hasani et al. (2022). "Closed-form Continuous-time Neural Networks." *Nature Machine Intelligence*.
- Lechner et al. (2020). "Neural Circuit Policies Enabling Auditable Autonomy." *Nature Machine Intelligence*.

### Bayesian Continual Learning

- Nguyen et al. (2018). "Variational Continual Learning." *ICLR* (VCL).
- Ritter et al. (2018). "Online Structured Laplace Approximations for Overcoming Catastrophic Forgetting." *NeurIPS*.
- Zeno et al. (2018). "Task Agnostic Continual Learning Using Online Variational Bayes." *NeurIPS Workshop*.

### Industrial Applications

- Downs & Vogel (1993). "A Plant-Wide Industrial Process Control Problem." *Computers & Chemical Engineering* (TEP).
- Yin et al. (2012). "A Comparison Study of Basic Data-Driven Fault Diagnosis Methods for TEP." *Journal of Process Control*.

### Related Work

**Continual Learning + RNNs:**
- Sodhani et al. (2020). "Toward Training Recurrent Neural Networks for Lifelong Learning." *Neural Computation*.

**Neural ODEs:**
- Chen et al. (2018). "Neural Ordinary Differential Equations." *NeurIPS*.

**Meta-Learning:**
- Finn et al. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation." *ICML*.

---

## Conclusion

This project combines **three cutting-edge research areas**:

1. **Continual Learning:** Mitigating catastrophic forgetting through replay, regularization, and Bayesian methods
2. **Continuous-time Neural Networks:** CfC/NCP for better temporal modeling and interpretability
3. **Industrial Process Monitoring:** Real-world application to Tennessee Eastman Process

**Key Hypothesis:**

> *Closed-form continuous-time networks with sparse wiring provide better representations for continual learning than standard RNNs, due to their multi-scale temporal dynamics, reduced parameter interference, and interpretable structure.*

**Expected Outcomes:**

1. Comprehensive benchmark of 21+ CL methods with CfC backbones
2. Bayesian CfC implementations (Laplace, VCL, Online)
3. Industrial TEP fault detection with <10% forgetting
4. Explainable AI analysis of learned representations

**Impact:**

- **Scientific:** New understanding of temporal inductive biases in CL
- **Engineering:** Practical methods for industrial continual learning
- **Safety:** Interpretable and uncertainty-aware fault detection

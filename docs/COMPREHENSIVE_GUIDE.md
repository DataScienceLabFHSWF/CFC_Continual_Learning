# CfC/LTC Networks for Continual Learning: Complete Guide

## Executive Summary

This project explores whether **Neural Circuit Policies (NCPs)** and **Liquid Time-Constant (LTC)** networks can help solve the **catastrophic forgetting** problem in continual learning. We're testing if brain-inspired sparse neural networks naturally retain old knowledge while learning new tasks.

---

## 1. The Problem: Catastrophic Forgetting

### What is Continual Learning?
Imagine teaching a neural network to recognize:
1. **Task 1:** Dogs vs. Cats
2. **Task 2:** Cars vs. Trucks  
3. **Task 3:** Planes vs. Boats

A human learns all three without forgetting. But standard neural networks **catastrophically forget**: when trained on Task 2, they forget Task 1. When trained on Task 3, they forget Tasks 1 and 2.

### Why Does This Happen?
Neural networks learn by adjusting weights using gradients. When learning Task 2:
- The gradients from Task 2 **overwrite** the weights that were important for Task 1.
- There's no mechanism to "protect" old knowledge.

### Current Solutions (and Their Limitations)

1. **Replay (e.g., Experience Replay, DER++):**
   - **Idea:** Store old data and replay it while learning new tasks.
   - **Problem:** Requires memory to store old examples. Not biologically plausible.

2. **Regularization (e.g., EWC, SI):**
   - **Idea:** Penalize changes to weights that were important for old tasks.
   - **Problem:** Slows down learning. Hard to balance old vs. new.

3. **Architecture-Based (e.g., Progressive Networks):**
   - **Idea:** Add new neurons for each task.
   - **Problem:** Network grows indefinitely. Not scalable.

**Our Question:** Can we design a network architecture that *naturally* resists forgetting without needing these explicit mechanisms?

---

## 2. Our Hypothesis: Why NCPs/LTCs Might Help

### What are Neural Circuit Policies (NCPs)?

NCPs are sparse recurrent neural networks inspired by the **C. elegans** worm's nervous system (302 neurons, fully mapped). Key properties:

1. **Sparse Connectivity:**
   - Standard RNN: Every neuron connects to every other neuron (dense).
   - NCP: Only ~20-30% of possible connections exist (sparse).
   - Structure: **Sensory → Inter → Command → Motor** (4 layers with specific roles).

2. **Liquid Time Constants (LTC):**
   - Each neuron has a "time constant" (τ, tau) that controls how fast it reacts.
   - Large τ → Slow neuron (like long-term memory).
   - Small τ → Fast neuron (like working memory).
   - τ is **adaptive**: learned during training, not fixed.

3. **Continuous-Time Dynamics:**
   - Standard RNN: Discrete updates at each timestep.
   - LTC/CfC: Models continuous differential equations (like physics simulations).

### Why This Might Resist Catastrophic Forgetting

We have **4 core hypotheses**:

#### Hypothesis 1: Modularity (Sparse Wiring)
**Claim:** Sparse connectivity creates functional "modules" that specialize for different features or tasks.

**Analogy:** Think of a company with departments (Sales, Engineering, HR). Each department has its own expertise. When you train Engineering on a new skill, Sales isn't affected.

**In NCPs:**
- Sensory neurons might specialize in detecting edges (Task 1) vs. colors (Task 2).
- Inter neurons might form separate "pathways" for different tasks.
- Because connections are sparse, gradients from Task 2 can't easily "reach" and overwrite Task 1's neurons.

**How We Test This:**
- Compare **AutoNCP** (structured sparsity) vs. **Random Sparse** (random connections) vs. **Dense** (all connections).
- **Prediction:** If AutoNCP > Random Sparse, the *structure* matters. If Random Sparse ≈ Dense, sparsity alone doesn't help.

#### Hypothesis 2: Temporal Stability (Liquid Time Constants)
**Claim:** Neurons with large τ act as "memory buffers" that resist rapid changes.

**Analogy:** Imagine a heavy flywheel vs. a light switch. The flywheel (large τ) is hard to speed up or slow down (stable). The switch (small τ) flips instantly (volatile).

**In LTCs:**
- Some neurons might learn large τ values and become "stable feature detectors" (e.g., "this is an edge").
- When Task 2 arrives, these stable neurons resist change because their large τ makes them slow to adapt.
- New neurons (small τ) quickly learn Task 2.

**How We Test This:**
- **Monitor τ values** during training. We expect to see:
  - A subset of neurons develop very large τ (>10).
  - These neurons' weights remain stable across tasks.
- **Ablation:** Fix τ to a constant. If performance drops, τ adaptation is critical.

#### Hypothesis 3: Gradient Isolation
**Claim:** Sparse wiring prevents gradients from propagating to all weights.

**Analogy:** In a fully connected graph, a rumor spreads to everyone. In a sparse graph, rumors only reach your neighbors.

**In NCPs:**
- Gradients from the output (Motor neurons) backpropagate through the network.
- In a dense network, **all** weights get updated.
- In NCP, gradients only flow through existing connections. Many weights are **shielded** from Task 2's gradients.

**How We Test This:**
- **Gradient Orthogonality:** Measure the cosine similarity between gradients from Task 1 and Task 2.
- **Prediction:** NCP gradients should be more orthogonal (closer to 0) than Dense gradients.

#### Hypothesis 4: Expressivity
**Claim:** Continuous-time dynamics allow richer representations with fewer parameters.

**Analogy:** A differential equation can describe complex motion (orbits, waves) with just a few parameters. A lookup table needs millions of entries.

**In CfC/LTC:**
- The ODE formulation might capture "dynamics" of a task (e.g., how features evolve over time) more efficiently than discrete RNNs.
- Smaller, more expressive models often generalize better.

**How We Test This:**
- Compare **CfC** (closed-form ODE) vs. **LTC** (numerical ODE) vs. **LSTM** (discrete RNN) with **same parameter count**.
- **Prediction:** CfC/LTC achieve higher accuracy or lower forgetting than LSTM.

---

## 3. Our Approach: What We're Building

### Datasets (Continual Learning Benchmarks)

1. **Split MNIST:**
   - **Tasks:** 5 tasks, each with 2 digits (0-1, 2-3, 4-5, 6-7, 8-9).
   - **Challenge:** Simple images, tests basic forgetting.

2. **Split CIFAR-10:**
   - **Tasks:** 5 tasks, each with 2 classes (airplane-car, bird-cat, etc.).
   - **Challenge:** Complex images, tests visual feature retention.

3. **Tennessee Eastman Process (TEP):**
   - **Tasks:** 22 fault types in a chemical plant (sequential time-series data).
   - **Challenge:** True temporal data, tests LTC's ODE capabilities.

### Architectures (Backbones)

We implement **multiple variants** to isolate what matters:

| Backbone | Wiring | Dynamics | Purpose |
|----------|--------|----------|---------|
| `mnistmlp` | Dense | Feedforward | Baseline (worst case) |
| `mnistcfc` | AutoNCP | CfC (closed-form) | **Main method** |
| `mnistltc` | AutoNCP | LTC (ODE solver) | Test if true ODE helps |
| `mnist_random_sparse` | Random | CfC | Test if NCP structure matters |
| `mnistcfc_dense` | Dense | CfC | Test if sparsity matters |

For CIFAR-10, we replace MLP with ResNet:
- `resnet18` (baseline)
- `cnn-cfc` (ResNet + CfC)

For TEP:
- `tepcfc` (designed for time-series)

### Continual Learning Methods

We test our backbones with standard CL algorithms:

1. **SGD (Lower Bound):** Pure catastrophic forgetting.
2. **Joint (Upper Bound):** Train on all tasks simultaneously (no forgetting, but cheating).
3. **Experience Replay (ER):** Store old examples, replay them.
4. **DER++:** Enhanced replay with distillation.
5. **EWC:** Regularize important weights.

**Key Experiment:** If `mnistcfc + SGD` beats `mnistmlp + ER`, it proves the architecture alone helps more than replay!

### Metrics

**Standard CL Metrics:**
- **Average Accuracy:** Final performance on all tasks.
- **Backward Transfer (BWT):** How much old tasks degrade. Lower (more negative) = more forgetting.
- **Forgetting:** Average accuracy drop on old tasks.

**New Mechanistic Metrics (we need to implement):**
1. **Representational Stability:** How much do neuron activations change after learning new tasks?
2. **Weight Change per Layer:** Which layers change most? (Expect: Motor changes, Inter stays stable)
3. **Gradient Interference:** Cosine similarity of gradients between tasks.
4. **Tau Distribution:** Histogram of time constants (expect bimodal: fast + slow neurons).

---

## 4. Implementation Details

### How We Handle Non-Sequential Data (MNIST, CIFAR)

**Problem:** CfC/LTC expect sequential input (like video or time-series), but MNIST is a single image.

**Solution (Chunking):**
- Split the 28×28 MNIST image into 28 rows.
- Feed each row as a "timestep" to the LTC.
- Sequence: Row 1 → Row 2 → ... → Row 28.
- Classification uses the final hidden state.

**Why This Works:**
- Humans don't see images all at once. Our eyes scan (saccades).
- This creates a "reading order" that the LTC can exploit.
- Even if the order is arbitrary, it gives the network temporal structure to work with.

### CfC vs. LTC

| Property | CfC | LTC |
|----------|-----|-----|
| Speed | Fast (1 step) | Slow (10-50 solver steps) |
| Accuracy | Good approximation | Exact ODE solution |
| Gradients | Stable (closed-form) | Can be noisy (through solver) |
| Use Case | Production, large-scale | Research, small benchmarks |

**When to Use Which:**
- **CfC:** MNIST, CIFAR (need speed for 198 experiments).
- **LTC:** TEP (true time-series, benefit from ODE), ablation studies.

### Wiring Configurations

**AutoNCP Wiring:**
```python
from ncps.wirings import AutoNCP
wiring = AutoNCP(units=256, output_size=128)
# Automatically creates Sensory/Inter/Command/Motor layers
# ~20-30% sparsity
```

**Random Sparse Wiring:**
```python
from ncps.wirings import Random
wiring = Random(units=256, output_dim=128, sparsity_level=0.7)
# 70% of connections removed randomly
# No structure, just sparsity
```

**Dense (FullyConnected):**
```python
from ncps.wirings import FullyConnected
wiring = FullyConnected(units=256, output_dim=128)
# All connections present (baseline)
```

---

## 5. Expected Results & Falsification

### If Our Hypotheses Are TRUE:

1. **Modularity (H1):**
   - `mnistcfc` (AutoNCP) > `mnist_random_sparse` > `mnistcfc_dense`
   - BWT (forgetting) improves with structure.

2. **Temporal Stability (H2):**
   - Tau monitoring shows bimodal distribution (fast + slow neurons).
   - LTC slightly outperforms CfC on TEP (benefits from true ODE).

3. **Gradient Isolation (H3):**
   - Gradient cosine similarity (Task 1 vs Task 2) closer to 0 for NCP than Dense.

4. **Expressivity (H4):**
   - CfC/LTC matches or beats LSTM with 50% fewer parameters.

### If Our Hypotheses Are FALSE:

1. **Modularity Fails:**
   - Random Sparse ≈ AutoNCP → Structure doesn't matter, only sparsity.
   - **Pivot:** Focus on sparsity alone, drop NCP-specific claims.

2. **Temporal Stability Fails:**
   - All tau → 0 (network ignores them) or uniform distribution.
   - **Pivot:** CfC is just a fancy RNN, drop LTC claims.

3. **Gradient Isolation Fails:**
   - Gradients still interfere heavily in NCP.
   - **Pivot:** Sparsity isn't enough, need explicit regularization.

4. **Expressivity Fails:**
   - LSTM > CfC/LTC at same parameter count.
   - **Pivot:** ODE overhead not worth it.

**This is good science:** We have clear predictions and ways to be wrong.

---

## 6. Current Status & Next Steps

### ✅ Completed:
1. Implemented CfC backbones for MNIST (`mnistcfc`).
2. Implemented LTC backbone (`mnistltc`).
3. Implemented Random Sparse baseline (`mnist_random_sparse`).
4. Set up benchmark infrastructure (198 experiments, parallel execution).
5. Documented hypotheses and literature review.

### 🔄 In Progress:
- Running baseline experiments (MNIST MLP + SGD/Joint/ER).

### ⏳ To Do:
1. **Implement CIFAR and TEP variants** of LTC and Random Sparse.
2. **Add Tau Monitoring** to track time constants during training.
3. **Implement Advanced Metrics** (representational stability, weight change, gradient interference).
4. **Run full experiment suite** (3-5 days of compute).
5. **Analyze results** and write paper.

---

## 7. How to Explain This to Peers

### Elevator Pitch (30 seconds):
"Neural networks forget old tasks when learning new ones (catastrophic forgetting). We're testing if brain-inspired sparse networks (NCPs) naturally resist this by creating isolated modules and stable memory neurons, reducing the need for storing old data."

### Technical Pitch (5 minutes):
"Current continual learning methods rely on replay (storing old data) or regularization (slowing down learning). We hypothesize that architectural priors—sparse connectivity and adaptive time constants from Neural Circuit Policies—can provide implicit regularization. We test this with 4 hypotheses: modularity (sparse wiring isolates tasks), temporal stability (slow neurons resist change), gradient isolation (sparse backprop protects weights), and expressivity (ODE dynamics are more efficient). We compare AutoNCP vs. Random Sparse vs. Dense on Split-MNIST/CIFAR/TEP to isolate what matters."

### For Non-Experts:
"Imagine you learned Spanish, then French. A normal neural network would forget Spanish (catastrophic forgetting). We're building networks inspired by worm brains—they have fewer connections but more specialized. Like how your brain has separate regions for language vs. math, our networks might naturally separate old vs. new knowledge without explicitly trying."

---

## 8. Key References

1. **Lechner et al. (2020):** "Neural Circuit Policies Enabling Auditable Autonomy" - Introduced NCPs.
2. **Hasani et al. (2021):** "Liquid Time-Constant Networks" - Introduced LTCs with adaptive tau.
3. **Hasani et al. (2022):** "Closed-form Continuous-time Neural Networks" - CfC for speed.
4. **Kirkpatrick et al. (2017):** "Overcoming Catastrophic Forgetting in Neural Networks" - EWC (baseline).
5. **Buzzega et al. (2020):** "Dark Experience for General Continual Learning" - DER++ (baseline).

---

## 9. Potential Impact

### If Successful:
- **Scientific:** First work connecting NCPs to continual learning. Opens new research direction.
- **Practical:** Reduced memory footprint (no replay buffer needed). Better on-device learning (phones, robots).
- **Biological:** Validates that brain-inspired sparsity isn't just efficiency—it's a *learning strategy*.

### If Unsuccessful:
- **Still Valuable:** Rigorous ablations showing what *doesn't* work. Rules out NCP hype.
- **Pivot:** Maybe NCPs shine on true temporal data (robotics, control), not vision.

---

This is a **well-designed experiment** with clear hypotheses, rigorous baselines, and potential for high-impact results. Let's build it! 🚀

# Hypotheses: NCPs for Continual Learning

This document outlines the theoretical hypotheses for why Neural Circuit Policies (NCPs) and Closed-form Continuous-time (CfC) networks might be beneficial for Continual Learning (CL), specifically in resisting catastrophic forgetting.

## 1. The Modularity Hypothesis

**Claim:** The sparse, structured connectivity of NCPs (inspired by *C. elegans*) creates functional modules that can specialize for different tasks or features, reducing interference between tasks.

**Rationale:**
- Standard MLPs and RNNs are fully connected (dense). A gradient update for Task A tends to alter weights used by Task B, causing forgetting.
- NCPs use `AutoNCP` wiring, which enforces sparsity and a specific flow of information (Sensory -> Inter -> Command -> Motor).
- This sparsity limits the "overlap" of weight updates between tasks.

**Prediction:**
- NCP-based backbones will show lower **Backward Transfer (BWT)** (less negative) compared to fully connected equivalents (e.g., standard LSTM or MLP), even without explicit CL strategies (like Replay).
- **Test:** Compare `MNISTcfc` (with AutoNCP) vs. `MNISTcfc` (with FullyConnected wiring) on Split-MNIST.

## 2. The Temporal Stability Hypothesis

**Claim:** The continuous-time dynamics and liquid time constants of CfC/LTC networks provide stable representations that are robust to noise and sudden shifts in distribution.

**Rationale:**
- CfC networks model hidden states as continuous trajectories $h(t)$.
- The time constant $\tau$ acts as a filter. If $\tau$ is large, the neuron is slow to change, acting as a memory buffer.
- In CL, this "sluggishness" might prevent the network from overfitting rapidly to the new task's distribution, implicitly regularizing the learning.

**Prediction:**
- CfC networks will retain performance on older tasks longer than standard RNNs (LSTM/GRU) during training on new tasks.
- **Test:** Monitor accuracy of Task 1 while training on Task 2, 3, 4... for CfC vs. LSTM.

## 3. The Gradient Isolation Hypothesis

**Claim:** The specific wiring of NCPs (Sensory -> Inter -> Command -> Motor) creates a hierarchy where lower-level features (Sensory/Inter) can be shared while higher-level control (Command/Motor) adapts, or vice-versa, depending on plasticity.

**Rationale:**
- In `AutoNCP`, connections are not all-to-all.
- Gradients from the output (Motor) might not propagate to all Inter neurons equally, preserving some "core" features.

**Prediction:**
- Weight analysis will show that certain subsets of weights in NCP remain stable across tasks, while others change significantly.

## 4. The Expressivity Hypothesis

**Claim:** Continuous-time dynamics allow CfC networks to learn complex temporal patterns with fewer parameters than discrete RNNs.

**Rationale:**
- CfC approximates the solution to a differential equation.
- This might allow it to capture the "dynamics" of a task (e.g., in TEP or sequential MNIST) more efficiently.
- Smaller, more expressive models often generalize better.

**Prediction:**
- CfC will achieve higher accuracy on complex temporal tasks (TEP) compared to LSTMs of similar parameter count.

---

## Experimental Validation Plan

To test these hypotheses, we perform the following ablations:

1.  **Wiring Ablation (Modularity):**
    *   `CfC + AutoNCP` vs. `CfC + FullyConnected`
    *   Expectation: AutoNCP has less forgetting.

2.  **Dynamics Ablation (Temporal Stability):**
    *   `CfC` vs. `RNN/LSTM` (with same wiring/size)
    *   Expectation: CfC retains history better.

3.  **Architecture Ablation (Sequence vs. Static):**
    *   `MNISTcfc` (Chunked) vs. `MNISTMLP` (Baseline)
    *   Expectation: Exploiting sequential nature of data (even if artificial) helps CL if the model can lock onto temporal invariants.

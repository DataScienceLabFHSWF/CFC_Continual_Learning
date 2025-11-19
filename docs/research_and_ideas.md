# Research & Ideas: Deep Dive into NCPs for Continual Learning

## 1. Hypothesis Verification & Falsifiability

We have established 4 core hypotheses. Here is a critical analysis of their verifiability and how we can rigorously test (and potentially falsify) them.

### H1: The Modularity Hypothesis (Sparsity)
*   **Claim:** NCP wiring creates functional modules that reduce interference.
*   **Critique:** Is it the *NCP structure* or just *sparsity*? Randomly pruning a network (like in PackNet) also helps CL.
*   **Falsification Test:**
    *   **Baseline:** "Random Sparse" Network. Create a network with the *same sparsity level* (same % of zeros) as AutoNCP, but with random connections.
    *   **Prediction:** If NCP > Random Sparse, then the *structure* matters. If NCP ~= Random Sparse, then only *sparsity* matters (H1 is partially falsified/refined).

### H2: The Temporal Stability Hypothesis (Liquid Time Constants)
*   **Claim:** Large time constants ($\tau$) act as a low-pass filter, resisting rapid changes (forgetting) from new tasks.
*   **Critique:** Does the network actually *learn* to use large $\tau$ for stable features? Or does it just learn to ignore $\tau$?
*   **Falsification Test:**
    *   **Metric:** Monitor the distribution of $\tau$ values throughout training.
    *   **Prediction:** We should see a subset of neurons developing very large $\tau$ (slow dynamics) that remain stable across tasks. If all $\tau$ converge to 0 (instant updates), the hypothesis is falsified.
    *   **Ablation:** "Frozen $\tau$". Fix $\tau$ to a constant value and see if CL performance drops.

### H3: The Gradient Isolation Hypothesis
*   **Claim:** Sparse wiring prevents gradients from propagating to all weights.
*   **Critique:** Gradients might still flow through shared "hub" neurons (Inter neurons).
*   **Falsification Test:**
    *   **Metric:** Gradient Orthogonality. Measure the cosine similarity between gradients of Task A and Task B.
    *   **Prediction:** NCP gradients should be more orthogonal (closer to 0) than Dense gradients.

---

## 2. Raw Liquid Time Constant (LTC) Networks

The user suggested exploring "raw" LTCs instead of just CfCs.

### CfC vs. LTC
*   **LTC (Liquid Time Constant):**
    *   Dynamics: $\frac{dh}{dt} = -\left[\frac{1}{\tau} + f(x)\right] \cdot h + A \cdot f(x)$
    *   Solved via: Numerical ODE solvers (Euler, Runge-Kutta) at each step.
    *   **Pros:** True continuous dynamics; highly expressive; theoretically grounded.
    *   **Cons:** Slow training (sequential solver steps); potential vanishing gradients through the solver.
*   **CfC (Closed-form Continuous):**
    *   Dynamics: Approximate closed-form solution.
    *   **Pros:** Fast (no solver loop); stable gradients.
    *   **Cons:** Approximation might lose some dynamic properties of the true ODE.

### Why try Raw LTCs for CL?
*   **Hypothesis:** The numerical solver in LTCs introduces a "depth" (steps in the solver) that might act as a stronger temporal buffer than the closed-form approximation.
*   **Experiment:**
    *   The `ncps` library supports `LTCCell`.
    *   We can swap `CfC` for `LTC` in `MNISTcfc` and `cnn_cfc`.
    *   **Trade-off:** Training will be 10-50x slower. We should test on smaller benchmarks (Split-MNIST) first.

---

## 3. New Experimental Ideas

### A. "Liquid" Regularization
*   **Idea:** If $\tau$ controls stability, we can explicitly regularize it.
*   **Method:** Add a loss term that penalizes changes to $\tau$ for important neurons (similar to EWC, but for time constants).
*   $$L_{reg} = \sum_i F_i (\tau_i^{new} - \tau_i^{old})^2$$
*   **Rationale:** Forcing the "speed" of processing to remain constant might preserve the function of the neuron better than just penalizing weights.

### B. Mixed-Dynamics Ensembles
*   **Idea:** Combine "Fast" and "Slow" networks.
*   **Method:** An ensemble of two networks:
    1.  **Fast Learner:** Small $\tau$, high learning rate (adapts to new task).
    2.  **Slow Learner:** Large $\tau$, low learning rate (preserves old knowledge).
*   **Connection:** This mimics the "Complementary Learning Systems" (CLS) theory (Hippocampus vs. Neocortex).

### C. Permuted MNIST with Temporal Structure
*   **Idea:** Standard Permuted MNIST destroys spatial structure.
*   **Method:** "Sequential Permuted MNIST". Treat the permuted image as a *sequence* of pixels (or chunks).
*   **Rationale:** This forces the network to rely *entirely* on temporal memory (since spatial structure is gone), making it a pure test of the CfC/LTC capability.

# Literature Review: NCPs, Liquid Networks, and Continual Learning

## 1. Neural Circuit Policies (NCPs) and Liquid Time-Constant (LTC) Networks

### Lechner et al. (2020) - "Neural Circuit Policies Enabling Auditable Autonomy"
- **Key Contribution:** Introduced NCPs, sparse recurrent neural networks inspired by the nervous system of *C. elegans*.
- **Architecture:** 4-layer structure: Sensory -> Inter -> Command -> Motor.
- **Properties:**
    - Sparse connectivity (reduces parameters).
    - Interpretable dynamics (auditable).
    - Robustness to noise and distribution shifts in reinforcement learning tasks (e.g., lane keeping).
- **Relevance to CL:** The robustness to distribution shifts is the key property we want to exploit for Continual Learning. If the network is robust to "noise" (which can be interpreted as the new task data interfering with old task knowledge), it might resist forgetting.

### Hasani et al. (2021) - "Liquid Time-constant Networks"
- **Key Contribution:** LTCs, continuous-time RNNs where the time constant $\tau$ depends on the input $x(t)$.
- **Dynamics:** $\frac{dh}{dt} = -\frac{h}{\tau(x)} + S(x, h)$
- **Relevance to CL:** "Liquid" dynamics allow the network to adapt its processing speed. This might allow it to "ignore" high-frequency noise or irrelevant updates from a new task, or conversely, adapt quickly when needed.

### Hasani et al. (2022) - "Closed-form Continuous-time Neural Networks" (CfC)
- **Key Contribution:** Solved the differential equation of LTCs in closed form.
- **Benefit:** Removes the need for slow ODE solvers during training/inference. Faster, more stable gradients.
- **Relevance to CL:** Makes it practical to train these networks on large CL benchmarks (MNIST, CIFAR) which was previously too slow with ODE solvers.

## 2. Continual Learning (CL) Context

### Catastrophic Forgetting
- Neural networks overwrite old knowledge when trained on new data.
- **Standard Solutions:**
    - **Replay:** Store old data (ER, DER++).
    - **Regularization:** Penalize changes to important weights (EWC, SI).
    - **Architecture:** Expand the network or isolate parameters (PNN, SupSup).

### Recurrent Continual Learning
- Most CL research focuses on Feedforward (MLP/CNN) networks.
- **Sodhani et al. (2020) - "Toward Training Recurrent Neural Networks for Lifelong Learning"**:
    - Showed that RNNs suffer from forgetting too.
    - Proposed GEM/A-GEM adaptations for RNNs.
- **Relevance:** Our work sits here. We are exploring if *better* RNN architectures (CfC/NCP) naturally handle CL better than LSTMs.

## 3. Intersection: Why NCPs for CL?

There is very little existing literature explicitly combining NCPs/LTCs with Continual Learning benchmarks (Split-MNIST, etc.).

**Potential Advantages:**
1.  **Sparsity:** As shown in "PackNet" or "PathNet", using subsets of weights is a valid CL strategy. NCPs enforce this structurally.
2.  **Stability:** The "boundedness" and stability of the ODE solution in CfC might prevent the "exploding gradients" or drastic weight shifts that cause catastrophic forgetting.
3.  **Causal Structure:** The Sensory->Motor flow might force the network to learn causal relationships which are often invariant across tasks (e.g., physics of the world), rather than spurious correlations that change per task.

## 4. References

1.  Lechner, M., et al. "Neural circuit policies enabling auditable autonomy." Nature Machine Intelligence 2.10 (2020): 642-652.
2.  Hasani, R., et al. "Liquid time-constant networks." AAAI (2021).
3.  Hasani, R., et al. "Closed-form continuous-time neural networks." Nature Machine Intelligence (2022).
4.  Sodhani, S., et al. "Toward training recurrent neural networks for lifelong learning." Neural computation 32.6 (2020): 1135-1173.

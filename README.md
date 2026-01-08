# CfC for Continual Learning# CFC_Continual_Learning

The WandB page for this project can be found here: [https://wandb.ai/fneubuerger/mammoth/]

Exploring **Closed-form Continuous-time (CfC)** networks and **Neural Circuit Policies (NCPs)** for continual learning scenarios.

> **Status (Jan 2026):** We have identified that "Pure" HOPE architectures are insufficient for Class-Incremental Learning due to the "Growing Head" problem. We are now benchmarking a **Hybrid HOPE** strategy (CfC + Titan Memory + Experience Replay) which has shown promising preliminary results (Retention restored from 0% -> 40%+).

## 🎯 Project OverviewThis project focuses on exploring the capabilities of Liquid Time Constant (LTC) neural networks for continual learning approaches, utilizing the Mammoth library.

LTC neural networks are known for their ability to adapt and learn from new data over time, making them suitable for tasks that require continual learning without forgetting previously acquired knowledge.

This project investigates whether the temporal dynamics and bounded behavior of CfC networks can mitigate catastrophic forgetting in continual learning. We integrate CfC architectures with the [Mammoth](https://github.com/aimagelab/mammoth) continual learning framework to benchmark against 25+ existing methods.

This project tries expore the capabilities of Liquid Time Constant (LTC) neural networks for continual learning using the Mammoth library. 

### Key QuestionsIt extends its scope to include the application of explainable AI methods to comprehend the decision-making processes of LTC models.

- Can continuous-time neural ODEs provide more stable representations for sequential task learning?

- Does the sparse, biologically-inspired AutoNCP wiring reduce interference between tasks?### Explainable AI Integration

- How do CfC networks compare to standard RNNs/LSTMs in continual learning benchmarks?

Understanding the decision-making process of neural networks is crucial, especially in real-world applications where transparency and interpretability are essential. The project employs methods of explainable AI to shed light on how CFC and LTC models arrive at their decisions. This not only enhances the interpretability of the models but also provides insights into the factors influencing their behavior.

## 🏗️ Architecture### Real-world Application



### CfC BackbonesTo ensure the practical relevance of the research, the project applies these models and explainability methods to industrial data. 

- **MNISTcfc**: Sequential MNIST processing with AutoNCP wiring (23K params vs 59K FC)By doing so, it aims to go beyond the typical benchmark datasets, addressing challenges and complexities inherent in real-world scenarios.

- **CNNCfC**: ResNet18 + CfC temporal processing for CIFAR (1.33M params)This real-world use case serves as a valuable testbed for assessing the models' effectiveness in environments with dynamic and evolving data.

- **TEPCfC**: Tennessee Eastman Process fault detection with continuous-time dynamics

The combination of continual learning, explainable AI, and real-world application sets this project apart, offering a comprehensive exploration of LTC neural networks and their applicability in industrial contexts.

### Datasets
- **seq-mnist**: 5 tasks (digit pairs 0-1, 2-3, ..., 8-9)
- **perm-mnist**: Permuted pixel sequences
- **rot-mnist**: Rotated MNIST variants
- **tennessee-eastman**: Industrial fault detection (22 fault classes, incremental learning)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/DataScienceLabFHSWF/CFC_Continual_Learning.git
cd CFC_Continual_Learning

# Create virtual environment with uv
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Install ncps in editable mode
uv pip install -e ncps/

# Setup secrets (copy template and fill in your WandB credentials)
cp .secrets.json.template .secrets.json
# Edit .secrets.json with your API keys
```

### Running Experiments

#### Quick Validation (1 epoch, ~10 minutes)
```bash
# Activate environment
source .venv/bin/activate
cd mammoth

# Test CfC on MNIST
python utils/main.py --dataset seq-mnist --model er --backbone mnistcfc \
  --n_epochs 1 --batch_size 32 --lr 0.03 --buffer_size 200

# Test CfC on CIFAR-10
python utils/main.py --dataset seq-cifar10 --model er --backbone cnn-cfc \
  --n_epochs 1 --batch_size 32 --lr 0.03 --buffer_size 200

# Test CfC on Tennessee Eastman Process
python utils/main.py --dataset tennessee-eastman --model er --backbone tepcfc \
  --n_epochs 1 --batch_size 32 --lr 0.001 --buffer_size 200
```

#### Paper Benchmarks (Full Suite, ~50-75 hours with 4 GPUs)
```bash
# Run all benchmarks for paper
./scripts/benchmarks/run_paper_benchmarks.sh --dataset all --max-parallel 4

# Run specific dataset
./scripts/benchmarks/run_paper_benchmarks.sh --dataset mnist --max-parallel 4

# Analyze results
python scripts/analysis/analyze_paper_results.py
```

See **[PAPER_BENCHMARKS.md](PAPER_BENCHMARKS.md)** for comprehensive benchmark guide.

#### Legacy Commands (for reference)
```bash
# Baseline (no continual learning)
python utils/main.py --dataset seq-mnist --model sgd --lr 0.03 --n_epochs 5 --batch_size 32

# EWC (Elastic Weight Consolidation)
python utils/main.py --dataset seq-mnist --model ewc_on --lr 0.03 --n_epochs 5 \
  --batch_size 32 --e_lambda 0.1 --gamma 1.0

# Experience Replay
python utils/main.py --dataset seq-mnist --model er --lr 0.03 --n_epochs 5 \
  --batch_size 32 --buffer_size 200
```

## 📊 Available Methods

Our Mammoth v2.0 integration includes 70+ continual learning methods:

### Replay-Based (Memory Buffer)
- **ER** (Experience Replay) - Simple reservoir sampling
- **DER** / **DER++** (Dark Experience Replay) - Stores logits + data
- **ER-ACE** (ER + Asymmetric Cross-Entropy) - Handles imbalanced classes
- **A-GEM** (Averaged Gradient Episodic Memory) - Gradient constraints
- **GEM** (Gradient Episodic Memory) - Task-specific constraints
- **GSS** (Gradient-based Sample Selection) - Smart buffer management
- **MER** (Meta-Experience Replay) - Meta-learning replay
- **GDumb** (Greedy Sampler + Dumb Learner) - Balanced buffer + retrain
- **iCaRL** (Incremental Classifier + Representation Learning) - Exemplars + distillation
- **HAL** (Hindsight Anchor Learning) - Anchor-based replay
- **X-DER** (eXtended DER) - Multi-loss replay variants
- **FDR** (Flattening Dark Replay) - Loss landscape smoothing

### Regularization-Based (Parameter Protection)
- **EWC** / **EWC-Online** (Elastic Weight Consolidation) - Fisher information weighting
- **SI** (Synaptic Intelligence) - Path-integral importance
- **LwF** / **LwF-MC** (Learning without Forgetting) - Knowledge distillation
- **MAS** (Memory Aware Synapses) - Output-based importance

### Architecture-Based (Dedicated Capacity)
- **PNN** (Progressive Neural Networks) - Task-specific columns
- **PackNet** - Dynamic pruning per task
- **HAT** (Hard Attention to Task) - Task-specific masks

### Class-Incremental Specialists
- **BiC** (Bias Correction) - Corrects classifier bias
- **LUCiR** (Learning Unified Classifier Incrementally) - Cosine classifier + distillation
- **SLCA** (Slow Learner with Classifier Alignment) - Staged learning

### Meta-Learning & Advanced
- **CODA-Prompt** - Prompt-based continual learning
- **DualPrompt** - Dual prompt pool (task-general + task-specific)
- **L2P** (Learning to Prompt) - Learnable prompt keys
- **RanPAC** - Random Path Selection
- **CLIP-based** methods - Vision-language continual learning

### Baselines
- **SGD** - Vanilla fine-tuning (catastrophic forgetting baseline)
- **Joint** - Train on all data at once (upper bound)

**Total**: 70+ methods across replay, regularization, architecture, prompting, and meta-learning paradigms.

See [Mammoth documentation](https://github.com/aimagelab/mammoth) for full method list and papers.

## 🧪 Experiments & Results

### Current Status (Mammoth v2.0 Migration Complete)

✅ **Completed**
- **Mammoth v2.0 Migration**: Successfully migrated from v1.x to v2.0 (70+ methods)
- **CfC Backbone Integration**: MNISTcfc, CNN-CfC, TEPcfc all working with v2.0 API
- **Dataset Wrappers Fixed**: TEP dataset properly integrated with `store_masked_loaders`
- **Gradient Issues Resolved**: Fixed hidden state handling in CfC/LSTM (no graph reuse errors)
- **Validation Suite**: Full validation across MNIST, CIFAR-10, TEP datasets
- **Benchmark Infrastructure**: Parallel execution system for paper-quality experiments

🔄 **Current Work**
- Running comprehensive paper benchmarks (30+ configurations × 3 seeds)
- Analyzing CfC vs standard backbone performance across 10+ CL methods
- Temporal dynamics analysis for TEP industrial fault detection

📋 **Planned**
- Bayesian continual learning with CfC (Laplace approximation, VCL)
- Uncertainty quantification for incremental fault detection
- Explainable AI integration for CfC decision-making analysis
- Extended benchmarks on additional temporal datasets

### Paper Benchmark Suite

Comprehensive experiments comparing CfC against standard backbones:

**Configuration**:
- **Datasets**: MNIST (5 tasks), CIFAR-10 (5 tasks), TEP (22 tasks)
- **Methods**: SGD, Joint, ER, DER++, ER-ACE, A-GEM, GEM, EWC, SI, LwF
- **Backbones**: MLP vs CfC (MNIST), ResNet18 vs CNN-CfC (CIFAR), LSTM vs CfC (TEP)
- **Seeds**: 3 runs per configuration
- **Total**: ~200 experiments

See **[PAPER_BENCHMARKS.md](PAPER_BENCHMARKS.md)** for details.

### Validation Results (1 epoch quick tests)

**MNIST + CfC**:
- Task 1: 99.62% accuracy ✅
- Excellent convergence with minimal epochs

**CIFAR-10 + CNN-CfC**:
- Task 1: 84.4% → Task 2: 52.85% Class-IL / 62.92% Task-IL
- Shows expected forgetting, validates continual learning setup

**Tennessee Eastman Process**:
- 22-task industrial fault detection
- Testing CfC temporal dynamics vs LSTM baseline

### Incremental vs Joint Evaluation

Universal benchmark methodology:
```python
convergence_ratio = incremental_accuracy / joint_accuracy
```
- **1.0** = perfect continual learning (no forgetting)
- **<1.0** = catastrophic forgetting present
- Measures how close incremental learning gets to joint training upper bound

## 📁 Project Structure

```
CFC_Continual_Learning/
├── mammoth/                      # Mammoth v2.0 CL framework
│   ├── backbone/                 # Network architectures
│   │   ├── MNISTcfc.py           # CfC for MNIST (23K params)
│   │   ├── cnn_cfc.py            # ResNet18 + CfC (1.33M params)
│   │   └── TEPcfc.py             # TEP fault detection (CfC + LSTM)
│   ├── datasets/                 # Dataset loaders
│   │   ├── seq_mnist.py          # Sequential MNIST (5 tasks)
│   │   ├── perm_mnist.py         # Permuted MNIST
│   │   ├── rot_mnist.py          # Rotated MNIST
│   │   └── tennessee_eastman.py  # TEP (22 fault classes)
│   ├── models/                   # 70+ CL methods (v2.0)
│   └── utils/                    # Training utilities
├── ncps/                         # Neural Circuit Policies library
├── scripts/                      # Organized executable scripts
│   ├── validation/               # Quick validation tests (1 epoch)
│   ├── benchmarks/               # Paper benchmarks (10+ epochs)
│   │   ├── run_paper_benchmarks.sh
│   │   └── benchmark_runner.py
│   └── analysis/                 # Result analysis
│       ├── analyze_paper_results.py
│       ├── interpretability_analysis.py
│       └── visualize_results.py
├── configs/                      # Experiment configurations
│   ├── paper_benchmarks.yaml     # Full paper benchmark config
│   └── validate_*.yaml           # Quick validation configs
├── docs/                         # Documentation
│   ├── BENCHMARK_SYSTEM.md
│   ├── MAMMOTH_VERSION.md
│   ├── QUICK_REFERENCE.md
│   └── VALIDATION_RESULTS.md
├── tests/                        # Test scripts
├── results/                      # Experiment results
│   ├── validation/               # Quick test results
│   ├── benchmarks/               # Full benchmark results
│   └── checkpoints/              # Model checkpoints
├── data/                         # Downloaded datasets (auto-created)
├── CL_pipeline.ipynb             # Main Jupyter notebook
├── PAPER_BENCHMARKS.md           # Paper benchmark guide
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── setup_env.sh                  # Environment setup script
```

## 🔬 Research Background

### CfC Networks
- **Paper**: ["Closed-form Continuous-time Neural Networks"](https://www.nature.com/articles/s42256-022-00556-7) (Nature Machine Intelligence, 2022)
- **Key Properties**:
  - Continuous-time dynamics via neural ODEs
  - Closed-form solution (no ODE solver needed)
  - Bounded, stable behavior

### NCPs (Neural Circuit Policies)
- **Paper**: ["Liquid Time-constant Networks"](https://ojs.aaai.org/index.php/AAAI/article/view/16936) (AAAI, 2021)
- **Key Properties**:
  - Sparse, biologically-inspired wiring (AutoNCP)
  - 60% parameter reduction vs fully-connected
  - Interpretable hidden state dynamics

### Why CfC for Continual Learning?
1. **Temporal Stability**: Bounded dynamics may reduce sensitivity to distribution shifts
2. **Sparse Connectivity**: AutoNCP wiring might localize task representations
3. **Continuous-Time**: Natural for processing sequential tasks with varying time scales
4. **Parameter Efficiency**: Fewer parameters could mean less interference

⚠️ **Note**: Original CfC/LTC papers make **NO explicit claims** about continual learning or catastrophic forgetting. This project is an exploratory investigation.

## 📈 Results Preview

### TEP Incremental Fault Detection
```
Incremental Learning: 67.03%
Joint Training:       94.57%
Forgetting Gap:       27.54%
Convergence Ratio:    0.709
```

### MNIST Sequential Tasks (SGD baseline, 1 epoch)
```
Task 1: 53.66%
Task 2: 48.49%
Task 3: 71.49%
Task 4: 72.13%
Task 5: 66.60%
Final Class-IL: 19.22% (severe forgetting)
```

## 🛠️ Development

### Adding New Backbones
1. Create backbone in `mammoth/backbone/`
2. Use `@register_backbone` decorator
3. Implement `forward()` with `returnt` parameter
4. Test with quick validation

### Adding New Datasets
See [Mammoth documentation](https://aimagelab.github.io/mammoth/datasets/build_a_dataset.html) for dataset integration.

### Testing
```bash
cd tests
python test_mnistcfc.py      # Test MNIST CfC backbone
python test_cnn_cfc.py        # Test CNN-CfC
python test_tep_data.py       # Test TEP data loading
```

## 📚 Documentation

### Quick Start
- **[README.md](README.md)** (this file) - Project overview and setup
- **[PAPER_BENCHMARKS.md](PAPER_BENCHMARKS.md)** - Comprehensive paper benchmark guide
  - Full experiment suite (200+ configurations)
  - Parallel execution on multiple GPUs
  - Result analysis and LaTeX table generation

### Scientific Background
- **[docs/SCIENTIFIC_BACKGROUND.md](docs/SCIENTIFIC_BACKGROUND.md)** (if exists) - Theory and background

### Technical Guides
- **[docs/MAMMOTH_VERSION.md](docs/MAMMOTH_VERSION.md)** - Mammoth v2.0 migration details
- **[docs/BENCHMARK_SYSTEM.md](docs/BENCHMARK_SYSTEM.md)** - Benchmarking system
- **[docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - Quick command reference
- **[docs/VALIDATION_RESULTS.md](docs/VALIDATION_RESULTS.md)** - Validation results

## 🤝 Contributing

This is a research project. Contributions, issues, and feature requests are welcome!

## 📚 References

```bibtex
@article{hasani2022closed,
  title={Closed-form continuous-time neural networks},
  author={Hasani, Ramin and Lechner, Mathias and Amini, Alexander and 
          Liebenwein, Lucas and Ray, Aaron and Tschaikowski, Max and 
          Teschl, Gerald and Rus, Daniela},
  journal={Nature Machine Intelligence},
  volume={4},
  number={11},
  pages={992--1003},
  year={2022},
  publisher={Nature Publishing Group UK London}
}

@inproceedings{hasani2021liquid,
  title={Liquid time-constant networks},
  author={Hasani, Ramin and Lechner, Mathias and Amini, Alexander and 
          Rus, Daniela and Grosu, Radu},
  booktitle={AAAI Conference on Artificial Intelligence},
  volume={35},
  number={9},
  pages={7657--7666},
  year={2021}
}

@article{boschini2022class,
  title={Class-Incremental Continual Learning into the eXtended DER-verse},
  author={Boschini, Matteo and Bonicelli, Lorenzo and Buzzega, Pietro and 
          Porrello, Angelo and Calderara, Simone},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```

## � Documentation

### Core Documentation
- **[SCIENTIFIC_BACKGROUND.md](SCIENTIFIC_BACKGROUND.md)** - Comprehensive scientific background covering:
  - Continual Learning theory and catastrophic forgetting
  - All 25+ CL methods (replay, regularization, architecture, meta-learning)
  - CfC/NCP network architecture and continuous-time dynamics
  - Bayesian continual learning (VCL, Laplace, Online Bayesian)
  - Tennessee Eastman Process application
  - Research contributions and open questions

### Practical Guides
- **[BENCHMARKING.md](BENCHMARKING.md)** - Detailed benchmarking guide:
  - Parallel GPU execution (6 experiments across 2x H200 NVL)
  - Configuration and usage
  - Performance estimates and troubleshooting

- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick command reference for running benchmarks

### Technical Information
- **[MAMMOTH_VERSION.md](MAMMOTH_VERSION.md)** - Mammoth version information:
  - Current v1.x (25 methods, customized with CfC)
  - Mammoth v2.0 (70+ methods, available for future migration)
  - Migration considerations and recommendations

## �📄 License

This project builds upon:
- [Mammoth](https://github.com/aimagelab/mammoth) - MIT License
- [ncps](https://github.com/mlech26l/ncps) - Apache 2.0 License

See individual LICENSE files in respective directories.

## 🎓 Academic Context

Developed at **FH South Westphalia - Data Science Lab**

For research inquiries: [DataScienceLabFHSWF](https://github.com/DataScienceLabFHSWF)

---

**WandB Project**: [https://wandb.ai/fneubuerger/mammoth](https://wandb.ai/fneubuerger/mammoth)

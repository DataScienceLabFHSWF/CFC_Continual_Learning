# CfC for Continual Learning# CFC_Continual_Learning

The WandB page for this project can be found here: [https://wandb.ai/fneubuerger/mammoth/]

Exploring **Closed-form Continuous-time (CfC)** networks and **Neural Circuit Policies (NCPs)** for continual learning scenarios.## About the Project



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

### Configuration

Create `.secrets.json` with your WandB credentials:
```json
{
  "wandb_api_key": "your-api-key",
  "wandb_entity": "your-entity",
  "wandb_project": "mammoth"
}
```

### Running Experiments

```bash
# Activate environment and load secrets
source setup_env.sh

# Run baseline (no continual learning)
cd mammoth
python utils/main.py --dataset seq-mnist --model sgd --lr 0.03 --n_epochs 5 --batch_size 32

# Run EWC (Elastic Weight Consolidation)
python utils/main.py --dataset seq-mnist --model ewc_on --lr 0.03 --n_epochs 5 \
  --batch_size 32 --e_lambda 0.1 --gamma 1.0

# Run Experience Replay
python utils/main.py --dataset seq-mnist --model er --lr 0.03 --n_epochs 5 \
  --batch_size 32 --buffer_size 200

# Tennessee Eastman Process experiment
cd ../tests
python tep_simple_test.py --num_faults 5 --epochs_per_fault 10 --joint_epochs 20
```

## 📊 Available Methods

Our Mammoth integration includes 25+ continual learning methods:

### Replay-Based
- **ER** (Experience Replay)
- **DER** (Dark Experience Replay)
- **DER++** (DER with additional features)
- **GDumb** (Greedy Sampler + Dumb Learner)
- **GSS** (Gradient-based Sample Selection)
- **HAL** (Hindsight Anchor Learning)
- **iCaRL** (Incremental Classifier and Representation Learning)
- **MER** (Meta-Experience Replay)
- **ER-ACE** (ER with Asymmetric Cross-Entropy)
- **X-DER** (eXtended DER)
- **FDR** (Flattening experience replay)

### Regularization-Based
- **EWC** (Elastic Weight Consolidation)
- **SI** (Synaptic Intelligence)
- **LwF** (Learning without Forgetting)
- **LwF-MC** (LwF Multi-Class)

### Architecture-Based
- **PNN** (Progressive Neural Networks)
- **RPC** (Representational Play with Continual)

### Knowledge Distillation
- **BiC** (Bias Correction)
- **LUCiR** (Learning a Unified Classifier Incrementally)

### Other
- **GEM** (Gradient Episodic Memory)
- **A-GEM** (Averaged GEM)
- **Joint** (joint training upper bound)
- **SGD** (baseline - no CL strategy)

## 🧪 Experiments

### Current Status

✅ **Completed**
- MNISTcfc backbone implementation and testing
- CNN-CfC (ResNet18 + CfC) implementation
- TEP dataset integration with incremental vs joint evaluation
- Baseline experiments showing catastrophic forgetting (67% incremental vs 95% joint on TEP)
- MNIST sanity checks with Mammoth framework

🔄 **In Progress**
- Systematic comparison of all Mammoth methods on seq-mnist
- CfC vs LSTM temporal dynamics analysis
- WandB integration with secret management

📋 **Planned**
- Bayesian continual learning with CfC (Laplace approximation, VCL, Online Bayesian)
- Uncertainty quantification for incremental fault detection
- Migration to new [Mammoth v2.0](https://github.com/aimagelab/mammoth) (70+ models)
- Explainable AI integration for CfC decision-making analysis

### Incremental vs Joint Evaluation

We use a universal benchmark methodology:
```python
convergence_ratio = incremental_accuracy / joint_accuracy
```
- **1.0** = perfect continual learning (no forgetting)
- **<1.0** = catastrophic forgetting present
- Measures how close incremental learning gets to the joint training upper bound

## 📁 Project Structure

```
CFC_Continual_Learning/
├── mammoth/              # Mammoth CL framework
│   ├── backbone/         # Network architectures
│   │   ├── MNISTcfc.py   # CfC for MNIST
│   │   ├── cnn_cfc.py    # ResNet18 + CfC
│   │   └── TEPcfc.py     # TEP fault detection
│   ├── datasets/         # Dataset loaders
│   │   ├── seq_mnist.py  # Sequential MNIST
│   │   ├── perm_mnist.py # Permuted MNIST
│   │   ├── rot_mnist.py  # Rotated MNIST
│   │   └── tennessee_eastman.py  # TEP incremental/joint
│   ├── models/           # 25+ CL methods
│   └── utils/            # Training utilities
├── ncps/                 # Neural Circuit Policies library
├── tests/                # Test scripts
│   ├── test_mnistcfc.py
│   ├── test_cnn_cfc.py
│   ├── tep_simple_test.py
│   └── test_tep_data.py
├── data/                 # Downloaded datasets (auto-created)
├── requirements.txt      # Python dependencies
├── setup_env.sh          # Environment setup script
└── .secrets.json.template # WandB credentials template
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

### Adding New Models
See [Mammoth documentation](https://aimagelab.github.io/mammoth/models/build_a_model.html) for creating new continual learning methods.

### Adding New Datasets
See [Mammoth documentation](https://aimagelab.github.io/mammoth/datasets/build_a_dataset.html) for dataset integration.

### Testing
```bash
cd tests
python test_mnistcfc.py      # Test MNIST CfC backbone
python test_cnn_cfc.py        # Test CNN-CfC
python test_tep_data.py       # Test TEP data loading
python tep_simple_test.py     # Run TEP incremental experiment
```

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

# Detailed Benchmark Report

Generated on: 2026-01-08 13:34:24.931610

## Dataset: seq-cifar10

| Model | Backbone | Samples | Class-IL (Mean ± Std) | Task-IL (Mean ± Std) |
|---|---|---|---|---|
| er | resnet18_vanilla | 2 | 60.79 ± 3.55 | 91.75 ± 0.40 |
| hope | hope | 9 | 13.39 ± 0.81 | 60.57 ± 2.70 |

## Dataset: seq-cifar100

| Model | Backbone | Samples | Class-IL (Mean ± Std) | Task-IL (Mean ± Std) |
|---|---|---|---|---|
| er | resnet18_vanilla | 2 | 80.75 ± 4.17 | 80.75 ± 4.17 |
| hope | hope | 1 | 3.05 | 21.57 |

## Dataset: seq-mnist

| Model | Backbone | Samples | Class-IL (Mean ± Std) | Task-IL (Mean ± Std) |
|---|---|---|---|---|
| er | mnistmlp | 2 | 78.47 ± 0.00 | 98.60 ± 0.00 |
| hope | hope | 2 | 15.90 ± 2.46 | 64.01 ± 7.86 |

## Dataset: tennessee-eastman

| Model | Backbone | Samples | Class-IL (Mean ± Std) | Task-IL (Mean ± Std) |
|---|---|---|---|---|
| derpp | tepcfc | 3 | 5.47 ± 1.68 | 100.00 ± 0.00 |
| derpp | teplstm | 3 | 4.76 ± 0.18 | 100.00 ± 0.00 |
| er | tepcfc | 5 | 10.76 ± 13.05 | 100.00 ± 0.00 |
| er | teplstm | 3 | 4.71 ± 0.20 | 100.00 ± 0.00 |
| ewc_on | tepcfc | 1 | 4.15 | 100.00 |
| ewc_on | teplstm | 1 | 4.40 | 100.00 |
| lwf | tepcfc | 1 | 4.20 | 100.00 |
| lwf | teplstm | 1 | 4.59 | 100.00 |
| sgd | tepcfc | 5 | 10.80 ± 12.26 | 100.00 ± 0.00 |
| sgd | teplstm | 3 | 4.23 ± 0.14 | 100.00 ± 0.00 |


Architecture Overview
=====================

This project evaluates continuous-time and sparsely wired neural backbones for continual learning.

Backbones included in the benchmark suite:

- ``mnistcfc``: CfC with AutoNCP wiring for Sequential MNIST
- ``mnistltc``: LTC implementation for Sequential MNIST
- ``mnist-random-sparse``: random sparse CfC for ablating wiring structure
- ``cnn-cfc``: ResNet-18 feature extractor with a CfC classification head for CIFAR-10
- ``cnn-ltc``: ResNet-18 feature extractor with an LTC classification head
- ``cnn-random-sparse``: random sparse CfC head for CIFAR-10
- ``tepcfc``: CfC backbone for Tennessee Eastman Process
- ``tepltc``: LTC backbone for Tennessee Eastman Process
- ``tep-random-sparse``: random sparse CfC backbone for TEP

The documentation also covers:

- how sparse AutoNCP wiring is used to reduce gradient interference
- why LTC/CfC continuous-time dynamics are expected to improve stability
- the role of replay buffers in the benchmark experiments

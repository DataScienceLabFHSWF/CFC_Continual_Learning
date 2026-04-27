Benchmark Runner
================

The paper benchmark runner executes the comprehensive experiment suite for this project.

Script location:

- ``scripts/benchmarks/run_paper_benchmarks.sh``

This script supports the following options:

- ``--dataset``: ``mnist``, ``cifar10``, ``tep``, ``all``
- ``--max-parallel``: maximum number of concurrent tmux sessions
- ``--dry-run``: print commands without executing
- ``--force``: rerun even if previous logs are present

The runner writes logs to ``paper_results/logs`` and results to ``paper_results``.

It also includes a robust completion check. Completed experiments are detected by any of these markers:

- ``Experiment completed:``
- ``wandb: Synced``
- ``Run history:``

This prevents stale or partial experiments from being incorrectly skipped.

If you want to rerun TEP experiments after fixing the dataset issue:

.. code-block:: bash

   make run-tep

To run ablation experiments for H1/H2:

.. code-block:: bash

   make run-ablations

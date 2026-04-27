Usage Guide
===========

This section describes how to set up the environment, run benchmarks, and use WandB logging.

Installation
------------

1. Create a Python virtual environment and activate it from the repository root:

   .. code-block:: bash

      python3 -m venv .venv
      source .venv/bin/activate

2. Install dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

3. Install the local ncps package:

   .. code-block:: bash

      pip install -e ncps/

4. Configure WandB credentials using the secrets template:

   .. code-block:: bash

      cp .secrets.json.template .secrets.json
      # Fill in wandb_api_key and other credentials

Running Benchmarks
------------------

The benchmark runner script is the primary entry point for paper experiments:

.. code-block:: bash

   ./scripts/benchmarks/run_paper_benchmarks.sh --dataset all --max-parallel 4

For a single dataset:

.. code-block:: bash

   ./scripts/benchmarks/run_paper_benchmarks.sh --dataset mnist --max-parallel 4

Logging to WandB
----------------

The benchmark script reads the WandB API key from ``.secrets.json`` and sends runs to the project ``mammoth`` under the entity in that file.

To run a single debug experiment with WandB logging:

.. code-block:: bash

   cd mammoth
   source ../.venv/bin/activate
   python utils/main.py --dataset seq-mnist --model er --backbone mnistcfc \
       --n_epochs 1 --batch_size 32 --lr 0.03 --buffer_size 200 \
       --wandb_entity fneubuerger --wandb_project mammoth --wandb_name "smoke_test"

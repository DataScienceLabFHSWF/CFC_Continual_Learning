Testing
=======

The repository includes a dedicated testing suite to catch regressions in the benchmark runner and backbone implementations.

To run the test suite:

.. code-block:: bash

   make test

New tests were added to detect:

- TEP backbone batch-size handling for ``tepltc`` and ``tep-random-sparse``
- Benchmark runner completion detection based on WandB sync markers

Test files:

- ``tests/test_tep_batch_handling.py``
- ``tests/test_benchmark_runner.py``

The tests are executed using Python's built-in unittest discovery.

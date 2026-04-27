import os
import subprocess
import tempfile
import unittest

class TestBenchmarkRunner(unittest.TestCase):
    def setUp(self):
        self.script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'benchmarks', 'run_paper_benchmarks.sh'))

    def test_skip_detection_regex(self):
        with open(self.script_path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn('Experiment completed:\\|wandb: Synced\\|Run history:', content,
            'Benchmark runner must detect completed experiments from WandB sync markers.')

    def test_skip_detection_function(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, 'dummy_seed0.log')
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('wandb: Synced run 12345\n')

            # Copy the benchmark runner script and patch LOG_DIR to use our temp directory.
            script_copy = os.path.join(tmpdir, 'run_paper_benchmarks.sh')
            with open(self.script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            script_content = script_content.replace('LOG_DIR="$RESULTS_DIR/logs"', f'LOG_DIR="{tmpdir}"')
            with open(script_copy, 'w', encoding='utf-8') as f:
                f.write(script_content)
            os.chmod(script_copy, 0o755)

            command = f'bash -lc "source {script_copy}; is_experiment_completed dummy 0; echo $?"'
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout.strip(), '0')

if __name__ == '__main__':
    unittest.main()

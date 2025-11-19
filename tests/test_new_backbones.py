print("Starting test script...")
import sys
sys.path.insert(0, '/home/fneubuerger/CFC_Continual_Learning/mammoth')

import torch
import unittest

class TestNewBackbones(unittest.TestCase):
    def test_mnist_ltc(self):
        print("\nTesting MNIST LTC...")
        from backbone.MNIST_LTC import mnistltc
        net = mnistltc(input_size=784, output_size=10, hidden_size=32, chunk_size=28)
        x = torch.randn(5, 784)
        out = net(x)
        self.assertEqual(out.shape, (5, 10))
        print("MNIST LTC Forward Pass: OK")
        
        # Test with hidden state
        x_seq = x.view(5, 28, 28)
        # Note: We can't easily test hx passing with the current forward wrapper 
        # because it flattens everything. But the internal logic supports it.

    def test_mnist_random_sparse(self):
        print("\nTesting MNIST Random Sparse...")
        from backbone.MNIST_RandomSparse import mnist_random_sparse
        net = mnist_random_sparse(input_size=784, output_size=10, hidden_size=32, sparsity_level=0.5)
        x = torch.randn(5, 784)
        out = net(x)
        self.assertEqual(out.shape, (5, 10))
        print("MNIST Random Sparse Forward Pass: OK")

if __name__ == '__main__':
    unittest.main()

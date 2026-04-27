import unittest
import torch
from backbone.TEP_LTC import tepltc
from backbone.TEP_RandomSparse import tep_random_sparse

class TestTEPBatches(unittest.TestCase):
    def setUp(self):
        self.seq_len = 50
        self.input_size = 52
        self.num_classes = 22

    def _forward_batch(self, model_cls, batch_sizes):
        model = model_cls(input_size=self.input_size, num_classes=self.num_classes, hidden_size=128)
        for bs in batch_sizes:
            x = torch.randn(bs, self.seq_len, self.input_size)
            out = model(x)
            self.assertEqual(out.shape, (bs, self.num_classes))

    def test_tepltc_batch_size_variation(self):
        self._forward_batch(tepltc, [32, 14, 32])

    def test_tep_random_sparse_batch_size_variation(self):
        self._forward_batch(tep_random_sparse, [32, 14, 32])

if __name__ == '__main__':
    unittest.main()

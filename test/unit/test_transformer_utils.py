import torch

from joeynmt.transformer import PositionalEncoding, LabelSmoothing
from .test_helpers import TensorTestCase


class TestTransformerUtils(TensorTestCase):

    def setUp(self):
        seed = 42
        torch.manual_seed(seed)

    def test_position_encoding(self):

        batch_size = 2
        max_time = 3
        emb_size = hidden_size = 12

        x = torch.zeros([batch_size, max_time, emb_size])
        pe = PositionalEncoding(emb_size, dropout=0.)
        output = pe(x)
        self.assertEqual(pe.pe.size(2), hidden_size)
        self.assertTensorAlmostEqual(output, pe.pe[:, :x.size(1)])

    def test_label_smoothing(self):

        size = 5
        padding_idx = 0
        smoothing = 0.4
        criterion = LabelSmoothing(size, padding_idx, smoothing)

        predict = torch.FloatTensor([[0.1, 0.1, 0.6, 0.1, 0.1],
                                     [0.1, 0.1, 0.6, 0.1, 0.1],
                                     [0.1, 0.1, 0.6, 0.1, 0.1]])

        v = criterion(predict.log(), torch.LongTensor([2, 1, 0]))

        self.assertTensorAlmostEqual(
            criterion.true_dist,
            torch.Tensor(
                [[0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
                 [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
                 [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]
            )
        )

        self.assertAlmostEqual(v.item(), 1.0663001537322998)

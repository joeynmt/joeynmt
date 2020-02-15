import torch

from joeynmt.encoders import TransformerEncoder
from .test_helpers import TensorTestCase


class TestTransformerEncoder(TensorTestCase):

    def setUp(self):
        self.emb_size = 12
        self.num_layers = 3
        self.hidden_size = 12
        self.ff_size = 24
        self.num_heads = 4
        self.dropout = 0.
        self.seed = 42
        torch.manual_seed(self.seed)

    def test_transformer_encoder_freeze(self):
        encoder = TransformerEncoder(freeze=True)
        for n, p in encoder.named_parameters():
            self.assertFalse(p.requires_grad)

    def test_transformer_encoder_forward(self):
        batch_size = 2
        time_dim = 4
        torch.manual_seed(self.seed)

        encoder = TransformerEncoder(
            hidden_size=self.hidden_size, ff_size=self.ff_size,
            num_layers=self.num_layers, num_heads=self.num_heads,
            dropout=self.dropout, emb_dropout=self.dropout)

        for p in encoder.parameters():
            torch.nn.init.uniform_(p, -0.5, 0.5)

        x = torch.rand(size=(batch_size, time_dim, self.emb_size))

        # no padding, no mask
        x_length = torch.Tensor([time_dim] * batch_size).int()
        mask = torch.ones([batch_size, time_dim, 1]) == 1

        output, hidden = encoder(x, x_length, mask)

        self.assertEqual(output.shape, torch.Size(
            [batch_size, time_dim, self.hidden_size]))
        self.assertEqual(hidden, None)

        output_target = torch.Tensor(
            [[[1.9728e-01, -1.2042e-01,  8.0998e-02,  1.3411e-03, -3.5960e-01,
               -5.2988e-01, -5.6056e-01, -3.5297e-01,  2.6680e-01,  2.8343e-01,
               -3.7342e-01, -5.9112e-03],
              [8.9687e-02, -1.2491e-01,  7.7809e-02, -1.3500e-03, -2.7002e-01,
               -4.7312e-01, -5.7981e-01, -4.1998e-01,  1.0457e-01,  2.9726e-01,
               -3.9461e-01,  8.1598e-02],
              [3.4988e-02, -1.3020e-01,  6.0043e-02,  2.7782e-02, -3.1483e-01,
               -3.8940e-01, -5.5557e-01, -5.9540e-01, -2.9808e-02,  3.1468e-01,
               -4.5809e-01,  4.3313e-03],
              [1.2234e-01, -1.3285e-01,  6.3068e-02, -2.3343e-02, -2.3519e-01,
               -4.0794e-01, -5.6063e-01, -5.5484e-01, -1.1272e-01,  3.0103e-01,
               -4.0983e-01,  3.3038e-02]],
             [[9.8597e-02, -1.2121e-01,  1.0718e-01, -2.2644e-02, -4.0282e-01,
               -4.2646e-01, -5.9981e-01, -3.7200e-01,  1.9538e-01,  2.7036e-01,
               -3.4072e-01, -1.7965e-03],
              [8.8470e-02, -1.2618e-01,  5.3351e-02, -1.8531e-02, -3.3834e-01,
               -4.9047e-01, -5.7063e-01, -4.9790e-01,  2.2070e-01,  3.3964e-01,
               -4.1604e-01,  2.3519e-02],
              [5.8373e-02, -1.2706e-01,  1.0598e-01,  9.3214e-05, -3.0493e-01,
               -4.4406e-01, -5.4723e-01, -5.2214e-01,  8.0374e-02,  2.6307e-01,
               -4.4571e-01,  8.7052e-02],
              [7.9567e-02, -1.2977e-01,  1.1731e-01,  2.6198e-02, -2.4024e-01,
               -4.2161e-01, -5.7604e-01, -7.3298e-01,  1.6698e-01,  3.1454e-01,
               -4.9189e-01,  2.4027e-02]]]
        )
        self.assertTensorAlmostEqual(output_target, output)

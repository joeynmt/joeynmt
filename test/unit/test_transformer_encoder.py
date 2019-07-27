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
        seed = 42
        torch.manual_seed(seed)

    def test_transformer_encoder_freeze(self):
        encoder = TransformerEncoder(freeze=True)
        for n, p in encoder.named_parameters():
            self.assertFalse(p.requires_grad)

    def test_transformer_encoder_forward(self):
        batch_size = 2
        time_dim = 4

        encoder = TransformerEncoder(
            hidden_size=self.hidden_size, ff_size=self.ff_size,
            num_layers=self.num_layers, num_heads=self.num_heads,
            dropout=self.dropout)

        x = torch.rand(size=(batch_size, time_dim, self.emb_size))
        # no padding, no mask
        x_length = torch.Tensor([time_dim] * batch_size).int()
        mask = torch.ones([batch_size, time_dim, 1]).byte()

        output, hidden = encoder(x, x_length, mask)

        self.assertEqual(output.shape, torch.Size(
            [batch_size, time_dim, self.hidden_size]))
        self.assertEqual(hidden, None)

        output_target = torch.Tensor(
            [[[-1.5273e+00, -6.2337e-01, 4.0339e-01, 8.1311e-01, -2.2023e-01,
               1.6188e+00, -1.6189e-01, -1.7497e+00, -4.5473e-01, 1.3068e+00,
               -2.5306e-01, 8.4824e-01],
              [-2.7095e-01, -1.0950e+00, -2.8882e-02, 6.5335e-01, -4.8153e-01,
               2.2214e+00, -4.6426e-01, 7.4417e-01, -1.7028e+00, 1.1036e+00,
               -2.5078e-01, -4.2845e-01],
              [-8.4540e-01, -2.4914e+00, 1.0237e+00, 7.6577e-01, 6.9849e-01,
               1.8340e-01, -1.3033e-01, -4.3476e-01, -9.5174e-01, 9.4170e-01,
               8.9802e-01, 3.4250e-01],
              [-1.7186e-01, -2.2072e+00, 2.0543e-01, 7.6266e-01, 6.5954e-01,
               1.7367e+00, -7.0193e-01, 1.0622e+00, -5.0359e-01, -9.1205e-01,
               -3.6165e-01, 4.3168e-01]],

             [[-1.5237e+00, -1.2591e+00, 1.5448e-01, 9.1942e-01, 4.4880e-02,
               1.1388e+00, -1.3668e+00, 2.9366e-01, -4.7291e-01, 1.9042e+00,
               -1.4779e-01, 3.1488e-01],
              [-1.4769e+00, -1.9001e+00, -2.0079e-03, 1.1815e+00, 1.9176e-01,
               1.1518e+00, -6.0833e-01, 7.1664e-01, -6.2820e-01, 1.3499e+00,
               -4.0527e-01, 4.2914e-01],
              [3.6206e-02, -1.3561e+00, 1.3219e+00, -9.6492e-01, 8.7291e-01,
               1.3253e+00, 1.9886e-01, 7.7454e-01, -1.1944e+00, -4.9269e-01,
               -1.3828e+00, 8.6121e-01],
              [9.8071e-02, -2.5441e+00, 6.0269e-01, 4.0902e-02, 7.5306e-01,
               9.4676e-01, -6.7460e-01, 1.5550e+00, -5.6898e-01, -6.6151e-02,
               -5.6029e-01, 4.1762e-01]]])
        self.assertTensorAlmostEqual(output_target, output)

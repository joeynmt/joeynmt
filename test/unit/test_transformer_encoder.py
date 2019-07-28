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

        x = torch.rand(size=(batch_size, time_dim, self.emb_size))
        # no padding, no mask
        x_length = torch.Tensor([time_dim] * batch_size).int()
        mask = torch.ones([batch_size, time_dim, 1]).byte()

        output, hidden = encoder(x, x_length, mask)

        self.assertEqual(output.shape, torch.Size(
            [batch_size, time_dim, self.hidden_size]))
        self.assertEqual(hidden, None)

        output_target = torch.Tensor(
            [[[-1.3269, -1.3628, 0.8064, 1.0445, -0.2099, 1.6897, -0.4917,
               0.3183, -1.1799, 1.1991, -0.7177, 0.2308],
              [0.3716, -1.6637, 0.9539, 0.6346, -0.2840, 1.8331, -0.3731,
               0.4118, -1.6991, 0.8021, -0.5767, -0.4105],
              [0.1198, -2.1717, 0.9120, 0.5721, 0.2346, 1.3326, -0.1806,
               0.8509, -1.7899, 0.4471, 0.0174, -0.3443],
              [-0.4092, -2.2140, 0.9928, 0.4462, 0.3266, 1.7515, -0.3511,
               0.4381, -1.2372, 0.4875, -0.5957, 0.3645]],
             [[-1.5191, -1.3312, 0.5885, 1.2696, 0.1950, 0.9546, -0.0913,
               0.2079, -1.1017, 1.7440, -0.8362, -0.0801],
              [-0.2515, -1.8600, 0.1885, 0.6976, 0.1687, 1.7393, -0.7378,
               0.9397, -1.1552, 1.2756, -0.4407, -0.5642],
              [0.1335, -1.8415, 1.0848, -0.0072, 0.2735, 1.2370, -0.1290,
               0.9355, -1.5464, 0.2892, -1.2988, 0.8693],
              [0.0336, -2.4393, 0.3923, -0.0821, 0.8518, 0.9173, -0.2132,
               1.6668, -0.9801, 0.5360, -0.4318, -0.2513]]])
        self.assertTensorAlmostEqual(output_target, output)

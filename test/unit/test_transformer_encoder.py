import torch

from joeynmt.encoders import TransformerEncoder, TransformerEncoderLayer
from .test_helpers import TensorTestCase


class TestRecurrentEncoder(TensorTestCase):

    def setUp(self):
        self.emb_size = 12
        self.num_layers = 3
        self.hidden_size = 12
        self.ff_size = 24
        self.num_heads = 4
        self.dropout = 0.
        seed = 42
        torch.manual_seed(seed)

    def test_position_encoding(self):
        encoder = TransformerEncoder(
            hidden_size=self.hidden_size, ff_size=self.ff_size,
            num_layers=self.num_layers, num_heads=self.num_heads)
        print(encoder.pe.pe.size())

        self.assertEqual(encoder.pe.pe.size(2), self.hidden_size)

    def test_transformer_freeze(self):
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

        print(output)

        self.assertEqual(output.shape, torch.Size(
            [batch_size, time_dim, self.hidden_size]))
        self.assertEqual(hidden, None)

        output_target = torch.Tensor(
            [[[-0.3265, 0.0122, -0.0341, 0.7526, 0.4451, 1.4627, -1.6215,
               -0.3470, -0.8464, 1.1855, -1.7348, 1.0522],
              [-0.1196, -0.3855, 0.1440, 0.6900, 0.7323, 0.5406, -1.4054,
               -0.1040, -0.6723, 1.0072, -2.0802, 1.6529],
              [-0.1832, -0.8174, 0.4648, 1.0698, 0.3134, 0.7789, -1.3107,
               -0.6078, -0.6801, 1.2221, -1.7309, 1.4810],
              [0.0373, -1.0230, 0.3748, 0.9031, 0.8080, 0.8910, -1.5453,
               -0.3477, -0.7343, 0.8726, -1.6531, 1.4166]],

             [[-0.6818, 0.4189, 0.2175, 0.9222, 1.0164, 1.1163, -1.5604,
               -0.2604, -1.2611, 0.4332, -1.5654, 1.2047],
              [0.3164, 0.2253, 0.1671, 0.9346, 0.5302, 0.8008, -1.1598,
               -0.6508, -1.4421, 0.4379, -1.8079, 1.6483],
              [-0.2856, -0.2017, 0.5398, 1.0087, 0.6515, 0.9520, -1.8903,
               -0.7624, -0.9946, 1.1984, -1.2558, 1.0398],
              [-0.1589, -1.0814, 0.2725, 1.2819, 0.8176, 0.4122, -1.6531,
               -0.5552, -0.2897, 1.0905, -1.4692, 1.3328]]])
        self.assertTensorAlmostEqual(output_target, output)

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
            [[[-0.4256, 0.1072, 0.0155, 0.7239, 0.4905, 1.3247, -1.5558,
               -0.3509, -0.9214, 1.2840, -1.7615, 1.0693],
              [-0.1745, -0.3157, 0.3251, 0.6532, 0.6617, 0.5979, -1.3970,
               -0.1533, -0.7765, 1.0856, -2.0777, 1.5712],
              [-0.1378, -0.8135, 0.5008, 1.0826, 0.3408, 0.8382, -1.2756,
               -0.6101, -0.7581, 1.0849, -1.7532, 1.5009],
              [-0.0490, -0.9483, 0.3841, 0.9291, 0.8505, 0.7881, -1.4945,
               -0.3270, -0.8172, 0.9365, -1.6795, 1.4272]],

             [[-0.6104, 0.4010, 0.1397, 0.8625, 1.0309, 1.0151, -1.6762,
               -0.2485, -1.0487, 0.5824, -1.6729, 1.2251],
              [0.2141, 0.2892, 0.2086, 0.9977, 0.4318, 0.9500, -1.1859,
               -0.6417, -1.5015, 0.5381, -1.7835, 1.4831],
              [-0.1667, -0.3265, 0.3493, 1.0160, 0.6305, 0.8068, -1.9274,
               -0.7047, -0.8184, 1.3463, -1.3166, 1.1116],
              [-0.1639, -1.1023, 0.2168, 1.2276, 0.7986, 0.3777, -1.6462,
               -0.4816, -0.2629, 1.3105, -1.4894, 1.2152]]])
        self.assertTensorAlmostEqual(output_target, output)

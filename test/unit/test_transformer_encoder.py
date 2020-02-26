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
            [[[0.1615, -0.1195, 0.0586, -0.0921, -0.3483, -0.3654, -0.6052,
               -0.3355, 0.3179, 0.2757, -0.2909, -0.0346],
              [0.1272, -0.1241, 0.0223, -0.1463, -0.3462, -0.1579, -0.5591,
               -0.6274, 0.1822, 0.3043, -0.3818, 0.0094],
              [0.0616, -0.1344, 0.0625, 0.0056, -0.2785, -0.4290, -0.5765,
               -0.5176, -0.0598, 0.3389, -0.5522, -0.1692],
              [0.1539, -0.1371, 0.0026, -0.0248, -0.0856, -0.3223, -0.5537,
               -0.3948, -0.2586, 0.2458, -0.2887, -0.0698]],
             [[0.1863, -0.1198, 0.1006, -0.0277, -0.3779, -0.3728, -0.6343,
               -0.3449, 0.2131, 0.2448, -0.3122, -0.1777],
              [0.0254, -0.1219, 0.0436, -0.0289, -0.2932, -0.2377, -0.6003,
               -0.5406, 0.2308, 0.3578, -0.3728, 0.0707],
              [0.1146, -0.1270, 0.1163, -0.0290, -0.3773, -0.3924, -0.5738,
               -0.6528, 0.1428, 0.3623, -0.4796, 0.0471],
              [0.0815, -0.1355, 0.1016, 0.0496, -0.3001, -0.4812, -0.5557,
               -0.6937, 0.1002, 0.2873, -0.4675, -0.1383]]]
        )
        self.assertTensorAlmostEqual(output_target, output)

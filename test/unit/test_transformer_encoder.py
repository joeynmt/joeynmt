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
        mask = torch.ones([batch_size, time_dim, 1]).byte()

        output, hidden = encoder(x, x_length, mask)

        self.assertEqual(output.shape, torch.Size(
            [batch_size, time_dim, self.hidden_size]))
        self.assertEqual(hidden, None)

        output_target = torch.Tensor(
            [[[0.1393, -0.1215, 0.0694, 0.0484, -0.3387, -0.4654, -0.5717,
               -0.4220, 0.1374, 0.2596, -0.3455, -0.0116],
              [0.1244, -0.1245, 0.1165, 0.0146, -0.2038, -0.5102, -0.6130,
               -0.3485, 0.0983, 0.3165, -0.4000, 0.0875],
              [0.1074, -0.1280, 0.0769, 0.1137, -0.3179, -0.3815, -0.5402,
               -0.6372, -0.0760, 0.3493, -0.4040, 0.0692],
              [0.2468, -0.1316, 0.0724, -0.0519, -0.2438, -0.4622, -0.5707,
               -0.4588, -0.0658, 0.4306, -0.4186, 0.0859]],
             [[0.2024, -0.1192, 0.1011, 0.1219, -0.3491, -0.5344, -0.5323,
               -0.3684, 0.2269, 0.2314, -0.4020, -0.0892],
              [0.0929, -0.1278, 0.0648, -0.0037, -0.3597, -0.4531, -0.5543,
               -0.5035, 0.1212, 0.4414, -0.5052, 0.0642],
              [0.0614, -0.1298, 0.1862, 0.0578, -0.2459, -0.4973, -0.5859,
               -0.4368, -0.0622, 0.2637, -0.5961, -0.0580],
              [0.0610, -0.1255, 0.0972, 0.1820, -0.2317, -0.4773, -0.5540,
               -0.5786, 0.1232, 0.3119, -0.4429, 0.0238]]]
        )
        self.assertTensorAlmostEqual(output_target, output)

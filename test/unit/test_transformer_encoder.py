from test.unit.test_helpers import TensorTestCase

import torch

from joeynmt.encoders import TransformerEncoder, TransformerEncoderLayer


class TestTransformerEncoder(TensorTestCase):
    def setUp(self):
        self.emb_size = 12
        self.num_layers = 3
        self.hidden_size = 12
        self.ff_size = 24
        self.num_heads = 4
        self.dropout = 0.0
        self.alpha = 1.0
        self.layer_norm = "pre"
        self.seed = 42
        torch.manual_seed(self.seed)

    def test_transformer_encoder_freeze(self):
        encoder = TransformerEncoder(freeze=True)
        for _, p in encoder.named_parameters():
            self.assertFalse(p.requires_grad)

    def test_transformer_encoder_forward(self):
        batch_size = 2
        time_dim = 4
        torch.manual_seed(self.seed)

        encoder = TransformerEncoder(
            hidden_size=self.hidden_size,
            ff_size=self.ff_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            emb_dropout=self.dropout,
            alpha=self.alpha,
            layer_norm=self.layer_norm,
        )

        for p in encoder.parameters():
            torch.nn.init.uniform_(p, -0.5, 0.5)

        x = torch.rand(size=(batch_size, time_dim, self.emb_size))

        # no padding, no mask
        x_length = torch.Tensor([time_dim] * batch_size).int()
        mask = torch.ones([batch_size, 1, time_dim]) == 1

        output, hidden = encoder(x, x_length, mask)

        self.assertEqual(
            output.shape, torch.Size([batch_size, time_dim, self.hidden_size])
        )
        self.assertEqual(hidden, None)

        output_target = torch.Tensor(
            [[[0.3490, -0.1221, 0.2713, 0.2581, -0.3874, -0.2854, -0.5425,
               0.0972, -0.2405, 0.4328, -0.6588, -0.4366],
              [0.2616, -0.1274, 0.3038, 0.2111, -0.3360, -0.1571, -0.5746,
               0.0589, -0.4680, 0.4528, -0.7163, -0.4043],
              [0.2023, -0.1331, 0.3057, 0.2621, -0.4199, -0.1331, -0.5279,
               -0.2394, -0.3170, 0.4415, -0.6931, -0.4674],
              [0.3028, -0.1369, 0.3293, 0.1930, -0.2990, -0.0809, -0.5313,
               0.0726, -0.5885, 0.4527, -0.6577, -0.4525]],
             [[0.2618, -0.1227, 0.3328, 0.1874, -0.5497, -0.1413, -0.5577,
               0.0270, -0.2582, 0.3828, -0.6330, -0.4598],
              [0.2601, -0.1278, 0.2718, 0.2001, -0.4562, -0.2642, -0.5329,
               -0.0919, -0.2477, 0.4362, -0.7344, -0.4522],
              [0.2495, -0.1288, 0.3594, 0.2098, -0.4271, -0.2422, -0.5324,
               -0.1033, -0.3689, 0.3338, -0.7045, -0.3822],
              [0.2717, -0.1330, 0.3880, 0.1990, -0.4064, -0.1314, -0.5246,
               -0.2744, -0.1479, 0.4265, -0.7675, -0.4789]]]
        )
        self.assertTensorAlmostEqual(output, output_target)

        for layer in encoder.layers:
            self.assertTrue(isinstance(layer, TransformerEncoderLayer))
            self.assertTrue(hasattr(layer, "src_src_att"))
            self.assertTrue(hasattr(layer, "feed_forward"))
            self.assertEqual(layer.alpha, self.alpha)
            self.assertEqual(layer._layer_norm_position, self.layer_norm)
            self.assertEqual(layer.size, self.hidden_size)
            self.assertEqual(
                layer.feed_forward.pwff_layer[0].in_features, self.hidden_size
            )
            self.assertEqual(
                layer.feed_forward.pwff_layer[0].out_features, self.ff_size
            )

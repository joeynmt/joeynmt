from test.unit.test_helpers import TensorTestCase

import torch

from joeynmt.encoders import TransformerEncoder


class TestTransformerEncoder(TensorTestCase):
    def setUp(self):
        self.emb_size = 12
        self.num_layers = 3
        self.hidden_size = 12
        self.ff_size = 24
        self.num_heads = 4
        self.dropout = 0.0
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
            [[[ 5.4377e-01,  2.7501e-01,  2.4093e-01, -3.8590e-01, -7.9208e-01,
                1.4016e-02,  4.1655e-01,  4.2080e-01,  3.2950e-02, -3.8715e-01,
                7.6244e-01,  1.6272e-01],
              [ 5.2966e-01,  2.7333e-01,  2.3766e-01, -3.9073e-01, -7.9986e-01,
                1.0278e-04,  4.1317e-01,  4.2015e-01,  3.7940e-02, -3.8140e-01,
                7.6265e-01,  1.6739e-01],
              [ 5.3515e-01,  2.7153e-01,  2.3073e-01, -3.9522e-01, -7.9593e-01,
                9.2096e-03,  4.1768e-01,  4.2067e-01,  3.9852e-02, -3.7603e-01,
                7.6170e-01,  1.6399e-01],
              [ 5.3237e-01,  2.6914e-01,  2.3058e-01, -3.9796e-01, -8.0102e-01,
                1.0312e-02,  4.1422e-01,  4.2089e-01,  4.5810e-02, -3.7641e-01,
                7.5984e-01,  1.6494e-01]],
             [[ 5.4239e-01,  2.7339e-01,  2.3396e-01, -3.9298e-01, -7.9124e-01,
                1.6456e-02,  4.1948e-01,  4.2083e-01,  3.5326e-02, -3.7985e-01,
                7.6206e-01,  1.6253e-01],
              [ 5.4096e-01,  2.7283e-01,  2.3809e-01, -3.9737e-01, -7.9893e-01,
                1.5527e-02,  4.1902e-01,  4.2081e-01,  3.6781e-02, -3.7226e-01,
                7.5965e-01,  1.6326e-01],
              [ 5.3700e-01,  2.7417e-01,  2.4116e-01, -3.9549e-01, -7.9980e-01,
                7.8887e-03,  4.1715e-01,  4.2013e-01,  3.4510e-02, -3.7332e-01,
                7.6094e-01,  1.6516e-01],
              [ 5.3332e-01,  2.6947e-01,  2.3117e-01, -3.9819e-01, -7.9948e-01,
                1.1057e-02,  4.1384e-01,  4.2066e-01,  4.4917e-02, -3.7743e-01,
                7.6026e-01,  1.6499e-01]]]
        )
        self.assertTensorAlmostEqual(output, output_target)

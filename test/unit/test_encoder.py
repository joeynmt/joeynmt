from test.unit.test_helpers import TensorTestCase

import torch
from torch.nn import GRU, LSTM

from joeynmt.encoders import RecurrentEncoder


class TestRecurrentEncoder(TensorTestCase):

    def setUp(self):
        self.emb_size = 10
        self.num_layers = 3
        self.hidden_size = 7
        seed = 42
        torch.manual_seed(seed)

    def test_recurrent_encoder_size(self):
        for bidirectional in [True, False]:
            directional_factor = 2 if bidirectional else 1
            encoder = RecurrentEncoder(
                hidden_size=self.hidden_size,
                emb_size=self.emb_size,
                num_layers=self.num_layers,
                bidirectional=bidirectional,
            )
            self.assertEqual(encoder.rnn.hidden_size, self.hidden_size)
            # output size is affected by bidirectionality
            self.assertEqual(encoder.output_size, self.hidden_size * directional_factor)
            self.assertEqual(encoder.rnn.bidirectional, bidirectional)

    def test_recurrent_encoder_type(self):
        valid_rnn_types = {"gru": GRU, "lstm": LSTM}
        for name, obj in valid_rnn_types.items():
            encoder = RecurrentEncoder(rnn_type=name)
            self.assertEqual(type(encoder.rnn), obj)

    def test_recurrent_input_dropout(self):
        drop_prob = 0.5
        encoder = RecurrentEncoder(dropout=drop_prob, emb_dropout=drop_prob)
        input_tensor = torch.Tensor([2, 3, 1, -1])
        encoder.train()
        dropped = encoder.emb_dropout(input=input_tensor)
        # eval switches off dropout
        encoder.eval()
        no_drop = encoder.emb_dropout(input=input_tensor)
        # when dropout is applied, remaining values are divided by drop_prob
        self.assertGreaterEqual((no_drop - (drop_prob * dropped)).abs().sum(), 0)

        drop_prob = 1.0
        encoder = RecurrentEncoder(dropout=drop_prob, emb_dropout=drop_prob)
        all_dropped = encoder.emb_dropout(input=input_tensor)
        self.assertEqual(all_dropped.sum(), 0)
        encoder.eval()
        none_dropped = encoder.emb_dropout(input=input_tensor)
        self.assertTensorEqual(no_drop, none_dropped)
        self.assertTensorEqual((no_drop - all_dropped), no_drop)

    def test_recurrent_freeze(self):
        encoder = RecurrentEncoder(freeze=True)
        for _, p in encoder.named_parameters():
            self.assertFalse(p.requires_grad)

    def test_recurrent_forward(self):
        time_dim = 4
        batch_size = 2
        bidirectional = True
        directions = 2 if bidirectional else 1
        encoder = RecurrentEncoder(
            emb_size=self.emb_size,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            bidirectional=bidirectional,
        )
        x = torch.rand(size=(batch_size, time_dim, self.emb_size))
        # no padding, no mask
        x_length = torch.Tensor([time_dim] * batch_size).int()
        mask = torch.ones_like(x)
        output, hidden = encoder(embed_src=x, src_length=x_length, mask=mask)
        self.assertEqual(
            output.shape,
            torch.Size([batch_size, time_dim, directions * self.hidden_size]),
        )
        self.assertEqual(
            hidden.shape,
            torch.Size([batch_size, directions * self.hidden_size]),
        )
        hidden_target = torch.Tensor([
            [
                0.1323,
                0.0125,
                0.2900,
                -0.0725,
                -0.0102,
                -0.4405,
                0.1226,
                -0.3333,
                -0.3186,
                -0.2411,
                0.1790,
                0.1281,
                0.0739,
                -0.0536,
            ],
            [
                0.1431,
                0.0085,
                0.2828,
                -0.0933,
                -0.0139,
                -0.4525,
                0.0946,
                -0.3279,
                -0.3001,
                -0.2223,
                0.2023,
                0.0708,
                0.0131,
                -0.0124,
            ],
        ])
        output_target = torch.Tensor([[
            [
                [
                    0.0041,
                    0.0324,
                    0.0846,
                    -0.0056,
                    0.0353,
                    -0.2528,
                    0.0289,
                    -0.3333,
                    -0.3186,
                    -0.2411,
                    0.1790,
                    0.1281,
                    0.0739,
                    -0.0536,
                ],
                [
                    0.0159,
                    0.0248,
                    0.1496,
                    -0.0176,
                    0.0457,
                    -0.3839,
                    0.0780,
                    -0.3137,
                    -0.2731,
                    -0.2310,
                    0.1866,
                    0.0758,
                    0.0366,
                    -0.0069,
                ],
                [
                    0.0656,
                    0.0168,
                    0.2182,
                    -0.0391,
                    0.0214,
                    -0.4389,
                    0.1100,
                    -0.2625,
                    -0.1970,
                    -0.2249,
                    0.1374,
                    0.0337,
                    0.0139,
                    0.0284,
                ],
                [
                    0.1323,
                    0.0125,
                    0.2900,
                    -0.0725,
                    -0.0102,
                    -0.4405,
                    0.1226,
                    -0.1649,
                    -0.1023,
                    -0.1823,
                    0.0712,
                    0.0039,
                    -0.0228,
                    0.0444,
                ],
            ],
            [
                [
                    0.0296,
                    0.0254,
                    0.1007,
                    -0.0225,
                    0.0207,
                    -0.2612,
                    0.0061,
                    -0.3279,
                    -0.3001,
                    -0.2223,
                    0.2023,
                    0.0708,
                    0.0131,
                    -0.0124,
                ],
                [
                    0.0306,
                    0.0096,
                    0.1566,
                    -0.0386,
                    0.0387,
                    -0.3958,
                    0.0556,
                    -0.3034,
                    -0.2701,
                    -0.2165,
                    0.2061,
                    0.0364,
                    -0.0012,
                    0.0184,
                ],
                [
                    0.0842,
                    0.0075,
                    0.2181,
                    -0.0696,
                    0.0121,
                    -0.4389,
                    0.0874,
                    -0.2432,
                    -0.1979,
                    -0.2168,
                    0.1519,
                    0.0066,
                    -0.0080,
                    0.0485,
                ],
                [
                    0.1431,
                    0.0085,
                    0.2828,
                    -0.0933,
                    -0.0139,
                    -0.4525,
                    0.0946,
                    -0.1608,
                    -0.1140,
                    -0.1646,
                    0.0796,
                    -0.0202,
                    -0.0207,
                    0.0379,
                ],
            ],
        ]])
        self.assertTensorAlmostEqual(hidden_target, hidden)
        self.assertTensorAlmostEqual(output_target, output)

import torch

from joeynmt.decoders import TransformerDecoder, TransformerDecoderLayer
from .test_helpers import TensorTestCase


class TestTransformerDecoder(TensorTestCase):

    def setUp(self):
        self.emb_size = 12
        self.num_layers = 3
        self.hidden_size = 12
        self.ff_size = 24
        self.num_heads = 4
        self.dropout = 0.
        seed = 42
        torch.manual_seed(seed)

    def test_transformer_decoder_freeze(self):
        decoder = TransformerDecoder(freeze=True)
        for n, p in decoder.named_parameters():
            self.assertFalse(p.requires_grad)

    def test_transformer_decoder_output_size(self):

        vocab_size = 11
        decoder = TransformerDecoder(
            num_layers=self.num_layers, num_heads=self.num_heads,
            hidden_size=self.hidden_size, ff_size=self.ff_size,
            dropout=self.dropout, vocab_size=vocab_size)

        if not hasattr(decoder, "output_size"):
            self.fail("Missing output_size property.")

        self.assertEqual(decoder.output_size, vocab_size)

    def test_transformer_decoder_forward(self):
        batch_size = 2
        src_time_dim = 4
        trg_time_dim = 5
        vocab_size = 7

        trg_embed = torch.rand(size=(batch_size, trg_time_dim, self.emb_size))

        decoder = TransformerDecoder(
            num_layers=self.num_layers, num_heads=self.num_heads,
            hidden_size=self.hidden_size, ff_size=self.ff_size,
            dropout=self.dropout, emb_dropout=self.dropout,
            vocab_size=vocab_size)

        encoder_output = torch.rand(
            size=(batch_size, src_time_dim, self.hidden_size))

        for p in decoder.parameters():
            torch.nn.init.uniform_(p, -0.5, 0.5)

        src_mask = torch.ones(size=(batch_size, 1, src_time_dim)) == 1
        trg_mask = torch.ones(size=(batch_size, trg_time_dim, 1)) == 1

        encoder_hidden = None  # unused
        decoder_hidden = None  # unused
        unrol_steps = None  # unused

        output, states, _, _ = decoder(
            trg_embed, encoder_output, encoder_hidden, src_mask, unrol_steps,
            decoder_hidden, trg_mask)

        output_target = torch.Tensor(
            [[[0.1946, 0.6144, -0.1925, -0.6967, 0.4466, -0.1085, 0.3400],
              [0.1857, 0.5558, -0.1314, -0.7783, 0.3980, -0.1736, 0.2347],
              [-0.0216, 0.3663, -0.2251, -0.5800, 0.2996, 0.0918, 0.2833],
              [0.0389, 0.4843, -0.1914, -0.6326, 0.3674, -0.0903, 0.2524],
              [0.0373, 0.3276, -0.2835, -0.6210, 0.2297, -0.0367, 0.1962]],
             [[0.0241, 0.4255, -0.2074, -0.6517, 0.3380, -0.0312, 0.2392],
              [0.1577, 0.4292, -0.1792, -0.7406, 0.2696, -0.1610, 0.2233],
              [0.0122, 0.4203, -0.2302, -0.6640, 0.2843, -0.0710, 0.2984],
              [0.0115, 0.3416, -0.2007, -0.6255, 0.2708, -0.0251, 0.2113],
              [0.0094, 0.4787, -0.1730, -0.6124, 0.4650, -0.0382, 0.1910]]]
        )
        self.assertEqual(output_target.shape, output.shape)
        self.assertTensorAlmostEqual(output_target, output)

        greedy_predictions = output.argmax(-1)
        expect_predictions = output_target.argmax(-1)
        self.assertTensorEqual(expect_predictions, greedy_predictions)

        states_target = torch.Tensor(
            [[[0.0491, 0.5322, 0.0327, -0.9208, -0.5646, -0.1138, 0.3416,
               -0.3235, 0.0350, -0.4339, 0.5837, 0.1022],
              [0.1838, 0.4832, -0.0498, -0.7803, -0.5348, -0.1162, 0.3667,
               -0.3076, -0.0842, -0.4287, 0.6334, 0.1872],
              [0.0910, 0.3801, 0.0451, -0.7478, -0.4655, -0.1040, 0.6660,
               -0.2871, 0.0544, -0.4561, 0.5823, 0.1653],
              [0.1064, 0.3970, -0.0691, -0.5924, -0.4410, -0.0984, 0.2759,
               -0.3108, -0.0127, -0.4857, 0.6074, 0.0979],
              [0.0424, 0.3607, -0.0287, -0.5379, -0.4454, -0.0892, 0.4730,
               -0.3021, -0.1303, -0.4889, 0.5257, 0.1394]],

             [[0.1459, 0.4663, 0.0316, -0.7014, -0.4267, -0.0985, 0.5141,
               -0.2743, -0.0897, -0.4771, 0.5795, 0.1014],
              [0.2450, 0.4507, 0.0958, -0.6684, -0.4726, -0.0926, 0.4593,
               -0.2969, -0.1612, -0.4224, 0.6054, 0.1698],
              [0.2137, 0.4132, 0.0327, -0.5304, -0.4519, -0.0934, 0.3898,
               -0.2846, -0.0077, -0.4928, 0.6087, 0.1249],
              [0.1752, 0.3687, 0.0479, -0.5960, -0.4000, -0.0952, 0.5159,
               -0.2926, -0.0668, -0.4628, 0.6031, 0.1711],
              [0.0396, 0.4577, -0.0789, -0.7109, -0.4049, -0.0989, 0.3596,
               -0.2966, 0.0044, -0.4571, 0.6315, 0.1103]]]
        )

        self.assertEqual(states_target.shape, states.shape)
        self.assertTensorAlmostEqual(states_target, states)

    def test_transformer_decoder_layers(self):

        vocab_size = 7

        decoder = TransformerDecoder(
            num_layers=self.num_layers, num_heads=self.num_heads,
            hidden_size=self.hidden_size, ff_size=self.ff_size,
            dropout=self.dropout, vocab_size=vocab_size)

        self.assertEqual(len(decoder.layers), self.num_layers)

        for layer in decoder.layers:
            self.assertTrue(isinstance(layer, TransformerDecoderLayer))
            self.assertTrue(hasattr(layer, "src_trg_att"))
            self.assertTrue(hasattr(layer, "trg_trg_att"))
            self.assertTrue(hasattr(layer, "feed_forward"))
            self.assertEqual(layer.size, self.hidden_size)
            self.assertEqual(
                layer.feed_forward.pwff_layer[0].in_features, self.hidden_size)
            self.assertEqual(
                layer.feed_forward.pwff_layer[0].out_features, self.ff_size)

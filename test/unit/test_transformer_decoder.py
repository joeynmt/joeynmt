from test.unit.test_helpers import TensorTestCase

import torch

from joeynmt.decoders import TransformerDecoder, TransformerDecoderLayer


class TestTransformerDecoder(TensorTestCase):
    def setUp(self):
        self.emb_size = 12
        self.num_layers = 3
        self.hidden_size = 12
        self.ff_size = 24
        self.num_heads = 4
        self.dropout = 0.0
        self.alpha = 1.0
        self.layer_norm = "pre"
        seed = 42
        torch.manual_seed(seed)

    def test_transformer_decoder_freeze(self):
        decoder = TransformerDecoder(freeze=True)
        for _, p in decoder.named_parameters():
            self.assertFalse(p.requires_grad)

    def test_transformer_decoder_output_size(self):
        vocab_size = 11
        decoder = TransformerDecoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            ff_size=self.ff_size,
            dropout=self.dropout,
            vocab_size=vocab_size,
            alpha=self.alpha,
            layer_norm=self.layer_norm,
        )

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
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            ff_size=self.ff_size,
            dropout=self.dropout,
            emb_dropout=self.dropout,
            vocab_size=vocab_size,
            alpha=self.alpha,
            layer_norm=self.layer_norm,
        )

        encoder_output = torch.rand(size=(batch_size, src_time_dim, self.hidden_size))

        for p in decoder.parameters():
            torch.nn.init.uniform_(p, -0.5, 0.5)

        src_mask = torch.ones(size=(batch_size, 1, src_time_dim)) == 1
        trg_mask = torch.ones(size=(batch_size, trg_time_dim, 1)) == 1

        encoder_hidden = None  # unused
        decoder_hidden = None  # unused
        unroll_steps = None  # unused

        output, states, _, _ = decoder(
            trg_embed,
            encoder_output,
            encoder_hidden,
            src_mask,
            unroll_steps,
            decoder_hidden,
            trg_mask,
        )
        output_target = torch.Tensor(
            [[[0.0018, 0.5140, -0.2714, 0.6730, -0.3272, -0.2436, -0.1376],
              [-0.0400, 0.4480, -0.3068, 0.5913, -0.3687, -0.2587, -0.1277],
              [0.1995, 0.6608, -0.4618, 0.5780, -0.5330, -0.3233, -0.0856],
              [0.1655, 0.6503, -0.4412, 0.6251, -0.4670, -0.3430, -0.0783],
              [0.2210, 0.7319, -0.4285, 0.6844, -0.5360, -0.4406, -0.0987]],
             [[0.1218, 0.5083, -0.3938, 0.5726, -0.3587, -0.2617, -0.0983],
              [0.0495, 0.4911, -0.4161, 0.5333, -0.4626, -0.2469, -0.1027],
              [0.1539, 0.6446, -0.4962, 0.5497, -0.6133, -0.3453, -0.1129],
              [0.1984, 0.6495, -0.4829, 0.5638, -0.5863, -0.3637, -0.0764],
              [0.2352, 0.6660, -0.4747, 0.6254, -0.4420, -0.3711, -0.0897]]]
        )
        self.assertEqual(output.shape, output_target.shape)
        self.assertTensorAlmostEqual(output, output_target)

        greedy_predictions = output.argmax(-1)
        expect_predictions = output_target.argmax(-1)
        self.assertTensorEqual(expect_predictions, greedy_predictions)

        states_target = torch.Tensor(
            [[[0.4662, 0.0845, 0.2270, 0.0267, -0.2532, 0.1923, -0.6067,
               0.1880, 0.5735, -0.9194, -0.2794, -0.4317],
              [0.4056, 0.1187, 0.2098, 0.0435, -0.1972, 0.1432, -0.5501,
               0.1697, 0.3971, -0.9948, -0.2548, -0.4318],
              [0.4677, 0.2305, 0.2056, 0.0427, -0.1987, 0.1159, -0.3307,
               0.0220, 0.5769, -1.1744, -0.2387, -0.4283],
              [0.5152, 0.2554, 0.1921, 0.0422, -0.2456, 0.1162, -0.3988,
               0.1587, 0.5830, -1.1201, -0.2563, -0.4229],
              [0.5651, 0.2904, 0.1921, 0.0356, -0.3046, 0.1850, -0.3590,
               0.1268, 0.6326, -1.1765, -0.2694, -0.4146]],
             [[0.5282, 0.1048, 0.2142, 0.0164, -0.0708, 0.1220, -0.4567,
               0.0373, 0.4419, -1.1140, -0.2499, -0.4250],
              [0.3235, 0.1469, 0.2270, 0.0310, -0.1900, 0.1714, -0.4207,
               0.1868, 0.4637, -1.0635, -0.2231, -0.4081],
              [0.3591, 0.1929, 0.2023, 0.0485, -0.2394, 0.1503, -0.3091,
               0.0547, 0.5323, -1.2670, -0.1985, -0.3994],
              [0.4043, 0.2479, 0.1993, 0.0393, -0.2170, 0.1816, -0.2761,
               0.0525, 0.5478, -1.1919, -0.1997, -0.4045],
              [0.6057, 0.2144, 0.1762, 0.0237, -0.2024, 0.1049, -0.3341,
               0.0957, 0.5612, -1.1979, -0.2359, -0.4053]]]
        )

        self.assertEqual(states.shape, states_target.shape)
        self.assertTensorAlmostEqual(states, states_target)

    def test_transformer_decoder_layers(self):
        vocab_size = 7

        decoder = TransformerDecoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            ff_size=self.ff_size,
            dropout=self.dropout,
            vocab_size=vocab_size,
            alpha=self.alpha,
            layer_norm=self.layer_norm,
        )

        self.assertEqual(len(decoder.layers), self.num_layers)

        for layer in decoder.layers:
            self.assertTrue(isinstance(layer, TransformerDecoderLayer))
            self.assertTrue(hasattr(layer, "src_trg_att"))
            self.assertTrue(hasattr(layer, "trg_trg_att"))
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

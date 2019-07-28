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
        self.seed = 42

    def test_transformer_decoder_freeze(self):
        torch.manual_seed(self.seed)
        encoder = TransformerDecoder(freeze=True)
        for n, p in encoder.named_parameters():
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
        torch.manual_seed(self.seed)
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

        src_mask = torch.ones(size=(batch_size, 1, src_time_dim)).byte()
        trg_mask = torch.ones(size=(batch_size, trg_time_dim, 1)).byte()

        encoder_hidden = None  # unused
        decoder_hidden = None  # unused
        unrol_steps = None  # unused

        output, states, _, _ = decoder(
            trg_embed, encoder_output, encoder_hidden, src_mask, unrol_steps,
            decoder_hidden, trg_mask)

        output_target = torch.Tensor(
            [[[0.1558, 0.6385, -0.1497, -0.7123, 0.4747, -0.0601, 0.3356],
              [-0.0911, 0.4325, -0.1746, -0.6000, 0.4458, -0.0274, 0.2255],
              [-0.1237, 0.2793, -0.2539, -0.5340, 0.3192, 0.1504, 0.2556],
              [0.0885, 0.5137, -0.1582, -0.5644, 0.3689, 0.0108, 0.2815],
              [-0.0503, 0.2164, -0.2965, -0.5538, 0.2544, 0.0695, 0.1583]],
             [[0.0743, 0.4331, -0.1659, -0.6578, 0.4237, -0.0831, 0.1560],
              [0.1377, 0.4527, -0.1841, -0.7375, 0.3034, -0.1648, 0.2502],
              [-0.0215, 0.2698, -0.2427, -0.6284, 0.2593, 0.0115, 0.2059],
              [0.1649, 0.4996, -0.1044, -0.7216, 0.4094, -0.1487, 0.1554],
              [0.0136, 0.4405, -0.0327, -0.5177, 0.5340, 0.0903, 0.1332]]]
        )
        self.assertEqual(output_target.shape, output.shape)
        self.assertTensorAlmostEqual(output_target, output)

        greedy_predictions = output.argmax(-1)
        expect_predictions = output_target.argmax(-1)
        self.assertTensorEqual(expect_predictions, greedy_predictions)

        states_target = torch.Tensor(
            [[[0.0722, 0.5148, -0.0432, -0.9611, -0.6177, -0.1057, 0.4504,
               -0.2954, 0.1108, -0.3940, 0.6351, 0.1466],
              [0.1613, 0.4742, -0.0723, -0.5786, -0.3589, -0.1142, 0.4140,
               -0.2926, 0.0495, -0.4598, 0.6545, 0.0892],
              [0.0732, 0.3670, 0.0154, -0.6322, -0.4113, -0.1073, 0.6896,
               -0.2840, 0.1473, -0.4557, 0.6109, 0.2081],
              [0.0039, 0.3626, -0.0303, -0.8609, -0.4719, -0.1039, 0.4347,
               -0.3181, -0.0041, -0.4597, 0.5354, 0.0810],
              [0.0099, 0.3415, 0.0182, -0.5281, -0.3664, -0.0905, 0.5646,
               -0.2928, -0.0322, -0.4822, 0.5512, 0.2129]],
             [[0.1118, 0.5019, 0.0513, -0.7626, -0.3657, -0.1020, 0.4257,
               -0.2956, -0.1217, -0.4480, 0.6188, 0.1316],
              [0.2548, 0.4837, 0.0969, -0.6586, -0.4360, -0.1071, 0.3801,
               -0.2971, -0.1570, -0.4711, 0.5877, 0.1356],
              [0.1731, 0.3887, 0.0751, -0.5601, -0.3748, -0.1085, 0.5684,
               -0.2884, -0.0492, -0.4868, 0.5905, 0.2180],
              [0.1646, 0.4525, 0.0220, -0.7763, -0.3819, -0.0990, 0.2693,
               -0.2852, -0.1779, -0.4833, 0.6045, 0.1714],
              [0.0174, 0.3602, -0.0373, -0.9379, -0.2617, -0.1205, 0.3944,
               -0.2926, 0.0495, -0.4650, 0.5970, 0.2099]]]
        )

        self.assertEqual(states_target.shape, states.shape)
        self.assertTensorAlmostEqual(states_target, states)

    def test_transformer_decoder_layers(self):

        torch.manual_seed(self.seed)
        batch_size = 2
        src_time_dim = 4
        trg_time_dim = 5
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

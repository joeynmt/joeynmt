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
            dropout=self.dropout, vocab_size=vocab_size)

        encoder_output = torch.rand(
            size=(batch_size, src_time_dim, self.hidden_size))

        src_mask = torch.ones(size=(batch_size, 1, src_time_dim)).byte()
        trg_mask = torch.ones(size=(batch_size, trg_time_dim, 1)).byte()

        encoder_hidden = None  # unused
        decoder_hidden = None  # unused
        unrol_steps = None  # unused

        output, states, _, _ = decoder(
            trg_embed, encoder_output, encoder_hidden, src_mask, unrol_steps,
            decoder_hidden, trg_mask)

        output_target = torch.Tensor(
            [[[-0.0805, 0.4592, 0.0718, 0.7900, 0.5230, 0.5067, -0.5715],
              [0.0711, 0.3738, -0.1151, 0.5634, 0.0394, 0.2720, -0.4201],
              [0.2361, 0.1380, -0.2817, 0.0559, 0.0591, 0.2231, -0.0882],
              [0.1779, 0.2605, -0.1604, -0.1684, 0.1802, 0.0476, -0.3675],
              [0.2059, 0.1267, -0.2322, -0.1361, 0.1820, -0.0788, -0.2393]],

             [[0.0538, 0.0175, -0.0042, 0.0384, 0.2151, 0.4149, -0.4311],
              [-0.0368, 0.1387, -0.3131, 0.3600, -0.1514, 0.4926, -0.2868],
              [0.1802, 0.0177, -0.4545, 0.2662, -0.3109, -0.0331, -0.0180],
              [0.3109, 0.2541, -0.3547, 0.0236, -0.3156, -0.0822, -0.0328],
              [0.3497, 0.2526, 0.1080, -0.5393, 0.2724, -0.4332, -0.3632]]])
        self.assertEqual(output_target.shape, output.shape)
        self.assertTensorAlmostEqual(output_target, output)

        greedy_predictions = output.argmax(-1)
        expect_predictions = output_target.argmax(-1)
        self.assertTensorEqual(expect_predictions, greedy_predictions)

        states_target = torch.Tensor(
            [[[-0.0755, 1.4055, -1.1602, 0.6213, 0.0544, 1.3840, -1.2356,
               -1.9077, -0.6345, 0.5314, 0.4973, 0.5196],
              [0.3196, 0.7497, -0.7922, -0.2416, 0.5386, 1.0843, -1.4864,
               -1.8824, -0.7546, 1.2005, 0.0748, 1.1896],
              [-0.3941, -0.1136, -0.9666, -0.4205, 0.2330, 0.7739, -0.4792,
               -2.0162, -0.4363, 1.6525, 0.5820, 1.5851],
              [-0.6153, -0.4550, -0.8141, -0.8289, 0.3393, 1.1795, -1.0093,
               -1.0871, -0.8108, 1.4794, 1.1199, 1.5025],
              [-0.6611, -0.6822, -0.7189, -0.6791, -0.1858, 1.5746, -0.5461,
               -1.0275, -0.9931, 1.5337, 1.3765, 1.0090]],

             [[-0.5529, 0.5892, -0.5661, -0.0163, -0.1006, 0.8997, -0.9661,
               -1.7280, -1.2770, 1.3293, 1.0589, 1.3298],
              [0.5863, 0.2046, -0.9396, -0.5605, -0.4051, 1.3006, -0.9817,
               -1.3750, -1.2850, 1.2806, 0.9258, 1.2487],
              [0.1955, -0.3549, -0.4581, -0.8584, 0.0424, 1.1371, -0.7769,
               -1.8383, -0.6448, 1.8183, 0.4338, 1.3043],
              [-0.0227, -0.8035, -0.5716, -0.9380, 0.3337, 1.2892, -0.7494,
               -1.5868, -0.5518, 1.5482, 0.5330, 1.5195],
              [-1.7046, -0.7190, 0.0613, -0.5847, 1.0075, 0.7987, -1.0774,
               -1.0810, -0.1800, 1.2212, 0.8317, 1.4263]]])

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
            self.assertTrue(hasattr(layer, "src_attn"))
            self.assertTrue(hasattr(layer, "self_attn"))
            self.assertTrue(hasattr(layer, "feed_forward"))
            self.assertEqual(layer.size, self.hidden_size)
            self.assertEqual(
                layer.feed_forward.layer[0].in_features, self.hidden_size)
            self.assertEqual(
                layer.feed_forward.layer[0].out_features, self.ff_size)






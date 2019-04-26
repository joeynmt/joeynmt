import torch

from joeynmt.encoders import TransformerEncoder, TransformerEncoderLayer
from joeynmt.decoders import TransformerDecoder, TransformerDecoderLayer
from joeynmt.embeddings import Embeddings
from joeynmt.transformer import PositionalEncoding
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

        self.assertEqual(decoder.output_size, self.hidden_size)

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
        trg_mask = torch.ones(size=(batch_size, trg_time_dim)).byte()

        encoder_hidden = None  # unused
        decoder_hidden = None  # unused
        unrol_steps = None  # unused

        output, states, _, _ = decoder(
            trg_embed, encoder_output, encoder_hidden, src_mask, unrol_steps,
            decoder_hidden, trg_mask)

        output_target = torch.Tensor(
            [[[-0.0912, 0.4453, 0.0645, 0.7826, 0.5222, 0.5410, -0.5620],
              [0.0705, 0.3444, -0.1202, 0.5674, 0.0220, 0.2931, -0.3922],
              [0.1913, 0.1230, -0.3112, 0.0528, -0.0142, 0.3447, -0.0859],
              [0.1729, 0.3278, -0.1322, -0.1608, 0.1408, 0.1450, -0.3974],
              [0.2156, 0.0971, -0.1856, -0.1555, 0.1234, -0.0181, -0.2090]],
             [[0.0428, -0.0047, -0.0143, 0.0289, 0.1937, 0.4086, -0.4228],
              [-0.0537, 0.0942, -0.2878, 0.3237, -0.1901, 0.5866, -0.2449],
              [0.1026, 0.0322, -0.4689, 0.3650, -0.2139, 0.0451, -0.1446],
              [0.3209, 0.2738, -0.3494, 0.0914, -0.4500, -0.0238, 0.0493],
              [0.2975, 0.2314, 0.0655, -0.4934, 0.3150, -0.4421, -0.4362]]])
        self.assertEqual(output_target.shape, output.shape)
        self.assertTensorAlmostEqual(output_target, output)

        greedy_predictions = output.argmax(-1)
        expect_predictions = output_target.argmax(-1)
        self.assertTensorEqual(expect_predictions, greedy_predictions)

        states_target = torch.Tensor(
            [[[-0.0450, 1.4198, -1.2109, 0.5739, 0.0457, 1.3679, -1.1827,
               -1.9142, -0.6402, 0.4976, 0.5654, 0.5227],
              [0.3766, 0.8111, -0.8096, -0.2991, 0.5932, 0.9996, -1.4182,
               -1.9407, -0.7295, 1.1780, 0.0536, 1.1849],
              [-0.1168, -0.0744, -1.0583, -0.5530, 0.3127, 0.6761, -0.4596,
               -2.0401, -0.5047, 1.5067, 0.6920, 1.6195],
              [-0.4432, -0.3576, -0.9009, -0.8535, 0.5247, 1.0817, -1.1015,
               -1.1339, -0.8226, 1.3436, 1.0464, 1.6169],
              [-0.6327, -0.5875, -0.6840, -0.7085, -0.2789, 1.6060, -0.4430,
               -1.0282, -1.1196, 1.4197, 1.4375, 1.0192]],
             [[-0.5521, 0.5805, -0.5400, -0.1065, -0.0830, 0.8747, -0.9759,
               -1.7261, -1.2477, 1.3271, 1.0926, 1.3564],
              [0.6915, 0.2433, -0.9995, -0.6288, -0.6628, 1.3260, -0.7713,
               -1.1132, -1.4784, 1.1578, 1.0425, 1.1930],
              [0.4736, -0.1917, -0.5887, -0.7041, -0.0716, 1.1858, -0.9477,
               -1.6205, -0.9747, 1.8926, 0.4323, 1.1146],
              [0.0826, -0.7669, -0.5557, -0.9644, 0.1891, 1.4200, -0.6509,
               -1.6918, -0.5047, 1.3717, 0.5127, 1.5583],
              [-1.5689, -0.6707, 0.0566, -0.5892, 1.0769, 0.7327, -1.2552,
               -1.0371, -0.2615, 1.3452, 0.8067, 1.3644]]])

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






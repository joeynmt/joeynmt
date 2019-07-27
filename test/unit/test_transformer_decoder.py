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
            [[[0.7319, 0.2398, 0.7555, -0.3612, 0.2586, -0.5062, -0.8719],
              [1.5938, 0.4141, 0.7507, -0.3174, 0.4038, -0.4154, 0.0530],
              [-0.0148, 0.8326, 1.0077, -0.4400, 0.0744, 0.0261, -0.3053],
              [0.4636, 0.6364, 0.6751, -0.4213, -0.1239, -0.1210, -0.4839],
              [0.5679, 0.9373, 0.0701, -0.6026, -0.0756, 0.4710, -0.1027]],
             [[1.5640, 0.3567, 0.2772, -0.5109, 0.1873, -0.2353, -0.0338],
              [0.8644, 0.2621, -0.1030, -0.3142, 0.1606, 0.2239, -0.2413],
              [-0.3464, 0.7669, 0.6012, -0.3077, 0.3643, 0.3625, 0.1739],
              [0.1830, 0.6607, 0.9520, -0.5541, 0.3045, -0.1209, -0.2509],
              [0.7647, 0.8228, 0.7677, -0.2894, 0.1310, 0.2233, -0.6086]]]
        )
        self.assertEqual(output_target.shape, output.shape)
        self.assertTensorAlmostEqual(output_target, output)

        greedy_predictions = output.argmax(-1)
        expect_predictions = output_target.argmax(-1)
        self.assertTensorEqual(expect_predictions, greedy_predictions)

        states_target = torch.Tensor(
            [[[-0.5224, -0.9710, 0.7492, 0.9617, -0.4224, -1.9238, -0.2384,
               1.5327, -0.8947, 0.3237, -0.0456, 1.4510],
              [-1.4013, 0.0576, 1.2996, 0.3296, -0.8640, 0.3529, 0.0307,
               1.2734, -1.9737, -0.2493, -0.1791, 1.3236],
              [-0.4058, -1.8827, 1.3899, -0.0570, -0.2892, -0.4645, 0.6091,
               1.3330, -1.3363, 0.6855, -0.6755, 1.0938],
              [-0.6014, -1.7154, 0.5887, 0.1663, -0.2477, 0.4614, 0.1013,
               1.8208, -1.6559, 0.3945, -0.5473, 1.2348],
              [-0.9119, -0.9595, 0.4894, -0.4672, -1.7896, 1.0663, 0.7225,
               1.6456, -1.2230, 0.3015, 0.5550, 0.5707]],
             [[-1.1442, 0.0034, 0.1221, -0.0236, -1.5197, 1.0301, 0.3933,
               1.1540, -1.7248, -0.3134, 0.4201, 1.6027],
              [1.0179, -0.2918, -0.2445, 0.0299, -0.9674, 1.0142, -1.0799,
               2.0954, -1.5961, -0.2321, -0.4484, 0.7029],
              [0.7394, -0.9846, 1.6384, 0.1900, -0.6118, -1.1268, 0.3840,
               1.2013, -1.6164, 1.1673, -0.6806, -0.3003],
              [0.1580, -1.5567, 1.3823, 0.2902, -0.4933, -1.2450, 0.5700,
               1.4647, -1.0593, 0.0221, -0.7817, 1.2488],
              [-1.3499, -1.2396, 1.7217, 0.1649, -0.5657, 0.2208, -0.6341,
               1.6481, -1.0808, 0.4684, -0.2090, 0.8553]]])

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






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

        src_mask = torch.ones(size=(batch_size, 1, src_time_dim)).byte()
        trg_mask = torch.ones(size=(batch_size, trg_time_dim, 1)).byte()

        encoder_hidden = None  # unused
        decoder_hidden = None  # unused
        unrol_steps = None  # unused

        output, states, _, _ = decoder(
            trg_embed, encoder_output, encoder_hidden, src_mask, unrol_steps,
            decoder_hidden, trg_mask)

        output_target = torch.Tensor(
            [[[2.0113, 0.0425, 0.4260, -0.3421, 0.4340, -0.5559, -0.6935],
              [0.0418, 0.5148, 0.8568, 0.0046, 0.1765, -0.2564, -0.0871],
              [0.0926, 0.9880, 0.8574, -0.4934, 0.0867, 0.2819, -0.5858],
              [0.6554, 0.6557, 0.4790, -0.5954, 0.0244, -0.0806, -0.3606],
              [0.6377, 0.9121, 0.1717, -0.5583, -0.1935, 0.3939, -0.4799]],
             [[1.0428, 0.8636, 0.3137, -0.5442, 0.1533, 0.2858, -0.2042],
              [0.7923, 0.6103, 0.1409, -0.4624, 0.2773, 0.1365, 0.3488],
              [-0.2321, 1.2025, 0.4752, -0.4935, 0.0221, 0.5474, -0.2908],
              [0.2524, 0.9604, 0.5765, -0.5036, 0.1403, 0.4644, -0.0231],
              [0.5573, 0.9994, 0.8173, -0.3908, 0.0746, 0.1735, -0.5141]]]
        )
        self.assertEqual(output_target.shape, output.shape)
        self.assertTensorAlmostEqual(output_target, output)

        greedy_predictions = output.argmax(-1)
        expect_predictions = output_target.argmax(-1)
        self.assertTensorEqual(expect_predictions, greedy_predictions)

        states_target = torch.Tensor(
            [[[-1.0498, 0.3209, 0.4614, 1.1255, -1.0564, -0.3808, -1.0052,
               1.3589, -1.4166, -0.2102, 0.0805, 1.7719],
              [0.7408, -0.6741, 0.6474, -0.6284, -0.1497, -0.2830, -0.0836,
               1.9189, -1.9979, 0.7107, -1.0687, 0.8675],
              [-0.5755, -1.8195, 1.3901, -0.2200, -0.8974, -0.5218, 0.3307,
               1.5591, -0.9079, 0.7822, -0.2442, 1.1242],
              [-0.5642, -1.2964, 0.4628, -0.1278, -1.1411, -0.0605, 0.5669,
               2.1694, -1.2394, -0.2823, 0.1745, 1.3381],
              [-1.3072, -1.4073, 0.6387, -0.0139, -1.2676, 1.0450, 0.1916,
               1.8136, -1.0857, 0.3759, 0.3656, 0.6514]],
             [[-0.7225, -0.4026, 0.4896, -0.5495, -1.9955, 1.1517, 0.3184,
               1.2419, -1.3340, 0.3528, 0.0663, 1.3834],
              [0.6585, -0.1221, 0.3871, -0.3421, -1.4653, 0.8979, 0.2325,
               1.9550, -1.8945, -0.3673, -0.5738, 0.6340],
              [0.6322, -1.5633, 1.0753, -0.3359, -1.1179, 0.2834, 0.2160,
               1.7932, -0.9702, 0.7798, -1.2423, 0.4497],
              [0.5172, -1.2970, 0.9692, -0.4986, -0.9930, 1.1269, 0.1179,
               1.3461, -1.4886, 0.4011, -1.1923, 0.9911],
              [-1.2188, -1.2929, 1.4350, -0.4339, -1.0394, 0.1496, 0.1445,
               1.7462, -0.9135, 0.3650, -0.1397, 1.1980]]]
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

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
            [[[ 0.1718,  0.5595, -0.1996, -0.6924,  0.4351, -0.0850,  0.2805],
              [ 0.0666,  0.4923, -0.1724, -0.6804,  0.3983, -0.1111,  0.2194],
              [-0.0315,  0.3673, -0.2320, -0.6100,  0.3019,  0.0422,  0.2514],
              [-0.0026,  0.3807, -0.2195, -0.6010,  0.3081, -0.0101,  0.2099],
              [-0.0172,  0.3384, -0.2853, -0.5799,  0.2470,  0.0312,  0.2518]],
             [[ 0.0284,  0.3918, -0.2010, -0.6472,  0.3646, -0.0296,  0.1791],
              [ 0.1017,  0.4387, -0.2031, -0.7084,  0.3051, -0.1354,  0.2511],
              [ 0.0155,  0.4274, -0.2061, -0.6702,  0.3085, -0.0617,  0.2830],
              [ 0.0227,  0.4067, -0.1697, -0.6463,  0.3277, -0.0423,  0.2333],
              [ 0.0133,  0.4409, -0.1186, -0.5694,  0.4450,  0.0290,  0.1643]]]
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

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
        )

        encoder_output = torch.rand(size=(batch_size, src_time_dim, self.hidden_size))

        for p in decoder.parameters():
            torch.nn.init.uniform_(p, -0.5, 0.5)

        src_mask = torch.ones(size=(batch_size, 1, src_time_dim)) == 1
        trg_mask = torch.ones(size=(batch_size, trg_time_dim, 1)) == 1

        encoder_hidden = None  # unused
        decoder_hidden = None  # unused
        unrol_steps = None  # unused

        output, states, _, _ = decoder(
            trg_embed,
            encoder_output,
            encoder_hidden,
            src_mask,
            unrol_steps,
            decoder_hidden,
            trg_mask,
        )
        output_target = torch.Tensor(
            [[[ 0.4870,  0.5005, -0.0708,  0.6948, -0.1007, -0.0685,  0.4173],
              [ 0.4871,  0.5007, -0.0712,  0.6945, -0.1011, -0.0682,  0.4174],
              [ 0.4873,  0.5006, -0.0712,  0.6945, -0.1012, -0.0684,  0.4175],
              [ 0.4871,  0.5007, -0.0711,  0.6946, -0.1011, -0.0686,  0.4176],
              [ 0.4871,  0.5006, -0.0709,  0.6948, -0.1010, -0.0687,  0.4175]],
             [[ 0.4930,  0.4617, -0.0458,  0.6986, -0.2207, -0.1896,  0.4968],
              [ 0.4932,  0.4619, -0.0462,  0.6982, -0.2211, -0.1892,  0.4969],
              [ 0.4932,  0.4620, -0.0464,  0.6980, -0.2214, -0.1891,  0.4971],
              [ 0.4932,  0.4620, -0.0463,  0.6981, -0.2215, -0.1894,  0.4971],
              [ 0.4926,  0.4620, -0.0455,  0.6990, -0.2212, -0.1913,  0.4973]]]
        )
        self.assertEqual(output.shape, output_target.shape)
        self.assertTensorAlmostEqual(output, output_target)

        greedy_predictions = output.argmax(-1)
        expect_predictions = output_target.argmax(-1)
        self.assertTensorEqual(expect_predictions, greedy_predictions)

        states_target = torch.Tensor(
            [[[ 0.1071,  0.2125,  0.3015, -0.6217,  0.3997,  0.5978, -0.5982,
               -0.1466,  0.5841, -0.5400, -0.3156, -0.3987],
              [ 0.1067,  0.2127,  0.3018, -0.6216,  0.3997,  0.5978, -0.5980,
               -0.1464,  0.5845, -0.5401, -0.3156, -0.3988],
              [ 0.1065,  0.2128,  0.3017, -0.6217,  0.3997,  0.5982, -0.5977,
               -0.1463,  0.5845, -0.5400, -0.3156, -0.3988],
              [ 0.1068,  0.2129,  0.3016, -0.6216,  0.3997,  0.5980, -0.5980,
               -0.1465,  0.5842, -0.5401, -0.3156, -0.3987],
              [ 0.1069,  0.2129,  0.3014, -0.6217,  0.3997,  0.5981, -0.5979,
               -0.1465,  0.5840, -0.5401, -0.3156, -0.3987]],
             [[-0.0481,  0.3615,  0.2356, -0.6513,  0.4025,  0.7624, -0.5016,
               -0.1411,  0.4480, -0.5101, -0.3082, -0.4036],
              [-0.0487,  0.3618,  0.2359, -0.6513,  0.4025,  0.7622, -0.5013,
               -0.1409,  0.4483, -0.5101, -0.3082, -0.4037],
              [-0.0491,  0.3620,  0.2360, -0.6513,  0.4025,  0.7622, -0.5011,
               -0.1409,  0.4484, -0.5101, -0.3082, -0.4037],
              [-0.0491,  0.3623,  0.2360, -0.6512,  0.4025,  0.7623, -0.5011,
               -0.1409,  0.4482, -0.5102, -0.3082, -0.4037],
              [-0.0476,  0.3629,  0.2348, -0.6508,  0.4025,  0.7624, -0.5021,
               -0.1414,  0.4464, -0.5110, -0.3082, -0.4035]]]
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
        )

        self.assertEqual(len(decoder.layers), self.num_layers)

        for layer in decoder.layers:
            self.assertTrue(isinstance(layer, TransformerDecoderLayer))
            self.assertTrue(hasattr(layer, "src_trg_att"))
            self.assertTrue(hasattr(layer, "trg_trg_att"))
            self.assertTrue(hasattr(layer, "feed_forward"))
            self.assertEqual(layer.size, self.hidden_size)
            self.assertEqual(
                layer.feed_forward.pwff_layer[0].in_features, self.hidden_size
            )
            self.assertEqual(
                layer.feed_forward.pwff_layer[0].out_features, self.ff_size
            )

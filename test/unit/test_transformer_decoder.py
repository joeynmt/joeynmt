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
            [[[0.1718, 0.5595, -0.1996, -0.6924, 0.4351, -0.0850, 0.2805],
              [0.0666, 0.4923, -0.1724, -0.6804, 0.3983, -0.1111, 0.2194],
              [-0.0315, 0.3673, -0.2320, -0.6100, 0.3019, 0.0422, 0.2514],
              [-0.0026, 0.3807, -0.2195, -0.6010, 0.3081, -0.0101, 0.2099],
              [-0.0172, 0.3384, -0.2853, -0.5799, 0.2470, 0.0312, 0.2518]],
             [[0.0284, 0.3918, -0.2010, -0.6472, 0.3646, -0.0296, 0.1791],
              [0.1017, 0.4387, -0.2031, -0.7084, 0.3051, -0.1354, 0.2511],
              [0.0155, 0.4274, -0.2061, -0.6702, 0.3085, -0.0617, 0.2830],
              [0.0227, 0.4067, -0.1697, -0.6463, 0.3277, -0.0423, 0.2333],
              [0.0133, 0.4409, -0.1186, -0.5694, 0.4450, 0.0290, 0.1643]]]
        )
        self.assertEqual(output_target.shape, output.shape)
        self.assertTensorAlmostEqual(output_target, output)

        greedy_predictions = output.argmax(-1)
        expect_predictions = output_target.argmax(-1)
        self.assertTensorEqual(expect_predictions, greedy_predictions)

        states_target = torch.Tensor(
            [[[3.7535e-02, 5.3508e-01, 4.9478e-02, -9.1961e-01, -5.3966e-01,
               -1.0065e-01, 4.3053e-01, -3.0671e-01, -1.2724e-02, -4.1879e-01,
               5.9625e-01, 1.1887e-01],
              [1.3837e-01, 4.6963e-01, -3.7059e-02, -6.8479e-01, -4.6042e-01,
               -1.0072e-01, 3.9374e-01, -3.0429e-01, -5.4203e-02, -4.3680e-01,
               6.4257e-01, 1.1424e-01],
              [1.0263e-01, 3.8331e-01, -2.5586e-02, -6.4478e-01, -4.5860e-01,
               -1.0590e-01, 5.8806e-01, -2.8856e-01, 1.1084e-02, -4.7479e-01,
               5.9094e-01, 1.6089e-01],
              [7.3408e-02, 3.7701e-01, -5.8783e-02, -6.2368e-01, -4.4201e-01,
               -1.0237e-01, 5.2556e-01, -3.0821e-01, -5.3345e-02, -4.5606e-01,
               5.8259e-01, 1.2531e-01],
              [4.1206e-02, 3.6129e-01, -1.2955e-02, -5.8638e-01, -4.6023e-01,
               -9.4267e-02, 5.5464e-01, -3.0029e-01, -3.3974e-02, -4.8347e-01,
               5.4088e-01, 1.2015e-01]],
             [[1.1017e-01, 4.7179e-01, 2.6402e-02, -7.2170e-01, -3.9778e-01,
               -1.0226e-01, 5.3498e-01, -2.8369e-01, -1.1081e-01, -4.6096e-01,
               5.9517e-01, 1.3531e-01],
              [2.1947e-01, 4.6407e-01, 8.4276e-02, -6.3263e-01, -4.4953e-01,
               -9.7334e-02, 4.0321e-01, -2.9893e-01, -1.0368e-01, -4.5760e-01,
               6.1378e-01, 1.3509e-01],
              [2.1437e-01, 4.1372e-01, 1.9859e-02, -5.7415e-01, -4.5025e-01,
               -9.8621e-02, 4.1182e-01, -2.8410e-01, -1.2729e-03, -4.8586e-01,
               6.2318e-01, 1.4731e-01],
              [1.9153e-01, 3.8401e-01, 2.6096e-02, -6.2339e-01, -4.0685e-01,
               -9.7387e-02, 4.1836e-01, -2.8648e-01, -1.7857e-02, -4.7678e-01,
               6.2907e-01, 1.7617e-01],
              [3.1713e-02, 3.7548e-01, -6.3005e-02, -7.9804e-01, -3.6541e-01,
               -1.0398e-01, 4.2991e-01, -2.9607e-01, 2.1376e-04, -4.5897e-01,
               6.1062e-01, 1.6142e-01]]]
        )

        self.assertEqual(states_target.shape, states.shape)
        self.assertTensorAlmostEqual(states_target, states)

    def test_transformer_decoder_forward_mask_type(self):
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

        with self.assertRaisesRegex(AssertionError, "src_mask has to be of type `BoolTensor`"):
            decoder(trg_embed, encoder_output, None, src_mask, None, None, trg_mask)

        src_mask = torch.ones(size=(batch_size, 1, src_time_dim)).bool()

        with self.assertRaisesRegex(AssertionError, "trg_mask has to be of type `BoolTensor`"):
            decoder(trg_embed, encoder_output, None, src_mask, None, None, trg_mask),

        trg_mask = torch.ones(size=(batch_size, trg_time_dim, 1)).bool()

        output, states, _, _ = decoder(
            trg_embed, encoder_output, None, src_mask, None, None, trg_mask)

        output_target = torch.Tensor(
            [[[0.1718, 0.5595, -0.1996, -0.6924, 0.4351, -0.0850, 0.2805],
              [0.0666, 0.4923, -0.1724, -0.6804, 0.3983, -0.1111, 0.2194],
              [-0.0315, 0.3673, -0.2320, -0.6100, 0.3019, 0.0422, 0.2514],
              [-0.0026, 0.3807, -0.2195, -0.6010, 0.3081, -0.0101, 0.2099],
              [-0.0172, 0.3384, -0.2853, -0.5799, 0.2470, 0.0312, 0.2518]],
             [[0.0284, 0.3918, -0.2010, -0.6472, 0.3646, -0.0296, 0.1791],
              [0.1017, 0.4387, -0.2031, -0.7084, 0.3051, -0.1354, 0.2511],
              [0.0155, 0.4274, -0.2061, -0.6702, 0.3085, -0.0617, 0.2830],
              [0.0227, 0.4067, -0.1697, -0.6463, 0.3277, -0.0423, 0.2333],
              [0.0133, 0.4409, -0.1186, -0.5694, 0.4450, 0.0290, 0.1643]]]
        )
        self.assertEqual(output_target.shape, output.shape)
        self.assertTensorAlmostEqual(output_target, output)


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

from torch.nn import GRU, LSTM
import torch

from joeynmt.decoders import RecurrentDecoder
from joeynmt.encoders import RecurrentEncoder
from .test_helpers import TensorTestCase


class TestRecurrentDecoder(TensorTestCase):

    def setUp(self):
        self.addTypeEqualityFunc(torch.Tensor,
                                 lambda x, y, msg: self.failureException(
                                     msg) if not torch.equal(x, y) else True)
        self.emb_size = 10
        self.num_layers = 3
        self.hidden_size = 7
        self.vocab_size = 5
        seed = 42
        torch.manual_seed(seed)

        bidi_encoder = RecurrentEncoder(emb_size=self.emb_size,
                                   num_layers=self.num_layers,
                                   hidden_size=self.hidden_size,
                                   bidirectional=True)
        uni_encoder = RecurrentEncoder(emb_size=self.emb_size,
                                   num_layers=self.num_layers,
                                   hidden_size=self.hidden_size,
                                   bidirectional=False)
        self.encoders = [uni_encoder, bidi_encoder]

    def test_recurrent_decoder_size(self):
        # test all combinations of bridge, input_feeding, encoder directions
        for encoder in self.encoders:
            for bridge in [True, False]:
                for input_feeding in [True, False]:
                    decoder = RecurrentDecoder(hidden_size=self.hidden_size,
                                               encoder=encoder,
                                               attention="bahdanau",
                                               emb_size=self.emb_size,
                                               vocab_size=self.vocab_size,
                                               num_layers=self.num_layers,
                                               bridge=bridge,
                                               input_feeding=input_feeding)
                    self.assertEqual(decoder.rnn.hidden_size, self.hidden_size)
                    self.assertEqual(decoder.att_vector_layer.out_features,
                                     self.hidden_size)
                    self.assertEqual(decoder.output_layer.out_features,
                                     self.vocab_size)
                    self.assertEqual(decoder.output_size, self.vocab_size)
                    self.assertEqual(decoder.rnn.bidirectional, False)

                    if bridge:
                        self.assertTrue(decoder.bridge)
                        self.assertTrue(hasattr(decoder, "bridge_layer"))
                        self.assertEqual(decoder.bridge_layer.out_features,
                                         self.hidden_size)
                    else:
                        self.assertFalse(decoder.bridge)
                        self.assertFalse(hasattr(decoder, "bridge_layer"))

                    if input_feeding:
                        self.assertEqual(decoder.rnn_input_size,
                                         self.emb_size + self.hidden_size)
                    else:
                        self.assertEqual(decoder.rnn_input_size, self.emb_size)

    def test_recurrent_decoder_type(self):
        valid_rnn_types = {"gru": GRU, "lstm": LSTM}
        for name, obj in valid_rnn_types.items():
            decoder = RecurrentDecoder(rnn_type=name,
                                       hidden_size=self.hidden_size,
                                       encoder=self.encoders[0],
                                       attention="bahdanau",
                                       emb_size=self.emb_size,
                                       vocab_size=self.vocab_size,
                                       num_layers=self.num_layers,
                                       bridge=False,
                                       input_feeding=False)
            self.assertEqual(type(decoder.rnn), obj)

    def test_recurrent_input_dropout(self):
        drop_prob = 0.5
        decoder = RecurrentDecoder(hidden_size=self.hidden_size,
                                   encoder=self.encoders[0],
                                   attention="bahdanau",
                                   emb_size=self.emb_size,
                                   vocab_size=self.vocab_size,
                                   num_layers=self.num_layers,
                                   bridge=False,
                                   input_feeding=False,
                                   dropout=drop_prob)
        input_tensor = torch.Tensor([2, 3, 1, -1])
        decoder.train()
        dropped = decoder.rnn_input_dropout(input=input_tensor)
        # eval switches off dropout
        decoder.eval()
        no_drop = decoder.rnn_input_dropout(input=input_tensor)
        # when dropout is applied, remaining values are divided by drop_prob
        self.assertGreaterEqual((no_drop - (drop_prob*dropped)).abs().sum(), 0)

        drop_prob = 1.0
        decoder = RecurrentDecoder(hidden_size=self.hidden_size,
                                   encoder=self.encoders[0],
                                   attention="bahdanau",
                                   emb_size=self.emb_size,
                                   vocab_size=self.vocab_size,
                                   num_layers=self.num_layers,
                                   bridge=False,
                                   input_feeding=False,
                                   dropout=drop_prob)
        all_dropped = decoder.rnn_input_dropout(input=input_tensor)
        self.assertEqual(all_dropped.sum(), 0)
        decoder.eval()
        none_dropped = decoder.rnn_input_dropout(input=input_tensor)
        self.assertTensorEqual(no_drop, none_dropped)
        self.assertTensorEqual((no_drop - all_dropped), no_drop)

    def test_recurrent_hidden_dropout(self):
        hidden_drop_prob = 0.5
        decoder = RecurrentDecoder(hidden_size=self.hidden_size,
                                   encoder=self.encoders[0],
                                   attention="bahdanau",
                                   emb_size=self.emb_size,
                                   vocab_size=self.vocab_size,
                                   num_layers=self.num_layers,
                                   bridge=False,
                                   input_feeding=False,
                                   hidden_dropout=hidden_drop_prob)
        input_tensor = torch.Tensor([2, 3, 1, -1])
        decoder.train()
        dropped = decoder.hidden_dropout(input=input_tensor)
        # eval switches off dropout
        decoder.eval()
        no_drop = decoder.hidden_dropout(input=input_tensor)
        # when dropout is applied, remaining values are divided by drop_prob
        self.assertGreaterEqual((no_drop -
                                 (hidden_drop_prob * dropped)).abs().sum(), 0)

        hidden_drop_prob = 1.0
        decoder = RecurrentDecoder(hidden_size=self.hidden_size,
                                   encoder=self.encoders[0],
                                   attention="bahdanau",
                                   emb_size=self.emb_size,
                                   vocab_size=self.vocab_size,
                                   num_layers=self.num_layers,
                                   bridge=False,
                                   input_feeding=False,
                                   hidden_dropout=hidden_drop_prob)
        all_dropped = decoder.hidden_dropout(input=input_tensor)
        self.assertEqual(all_dropped.sum(), 0)
        decoder.eval()
        none_dropped = decoder.hidden_dropout(input=input_tensor)
        self.assertTensorEqual(no_drop, none_dropped)
        self.assertTensorEqual((no_drop - all_dropped), no_drop)

    def test_recurrent_freeze(self):
        decoder = RecurrentDecoder(hidden_size=self.hidden_size,
                                   encoder=self.encoders[0],
                                   attention="bahdanau",
                                   emb_size=self.emb_size,
                                   vocab_size=self.vocab_size,
                                   num_layers=self.num_layers,
                                   bridge=False,
                                   input_feeding=False,
                                   freeze=True)
        for n, p in decoder.named_parameters():
            self.assertFalse(p.requires_grad)

    def test_recurrent_forward(self):
        time_dim = 4
        batch_size = 2
        # make sure the outputs match the targets
        decoder = RecurrentDecoder(hidden_size=self.hidden_size,
                                   encoder=self.encoders[0],
                                   attention="bahdanau",
                                   emb_size=self.emb_size,
                                   vocab_size=self.vocab_size,
                                   num_layers=self.num_layers,
                                   bridge=False,
                                   input_feeding=False)
        encoder_states = torch.rand(size=(batch_size, time_dim,
                                          self.encoders[0].output_size))
        trg_inputs = torch.ones(size=(batch_size, time_dim, self.emb_size))
        # no padding, no mask
        #x_length = torch.Tensor([time_dim]*batch_size).int()
        mask = torch.ones(size=(batch_size, 1, time_dim)).byte()
        output, hidden, att_probs, att_vectors = decoder(
            trg_inputs, encoder_hidden=encoder_states[:, -1, :],
            encoder_output=encoder_states, src_mask=mask, unrol_steps=time_dim,
            hidden=None, prev_att_vector=None)
        self.assertEqual(output.shape, torch.Size(
            [batch_size, time_dim, self.vocab_size]))
        self.assertEqual(hidden.shape, torch.Size(
            [self.num_layers, batch_size, self.hidden_size]))
        self.assertEqual(att_probs.shape, torch.Size(
            [batch_size, time_dim, time_dim]))
        self.assertEqual(att_vectors.shape, torch.Size(
            [batch_size, time_dim, self.hidden_size]))
        hidden_target = torch.Tensor(
            [[[0.5977, -0.2173, 0.0900, 0.8608, -0.3638, 0.5332, -0.5538],
              [0.5977, -0.2173, 0.0900, 0.8608, -0.3638, 0.5332, -0.5538]],

             [[-0.2767, 0.4492, -0.0656, -0.2800, 0.2594, 0.1410, 0.0101],
              [-0.2767, 0.4492, -0.0656, -0.2800, 0.2594, 0.1410, 0.0101]],

             [[0.2118, 0.2190, -0.0875, 0.2177, -0.0771, -0.1014, 0.0055],
              [0.2118, 0.2190, -0.0875, 0.2177, -0.0771, -0.1014, 0.0055]]])
        output_target = torch.Tensor(
            [[[-0.2888, 0.1992, -0.1638, 0.1031, 0.3977],
              [-0.2917, 0.1922, -0.1755, 0.1093, 0.3963],
              [-0.2938, 0.1892, -0.1868, 0.1132, 0.3986],
              [-0.2946, 0.1885, -0.1964, 0.1155, 0.4019]],

             [[-0.3103, 0.2316, -0.1540, 0.0833, 0.4444],
              [-0.3133, 0.2251, -0.1653, 0.0898, 0.4433],
              [-0.3153, 0.2223, -0.1763, 0.0939, 0.4458],
              [-0.3160, 0.2217, -0.1856, 0.0963, 0.4492]]])
        att_vectors_target = torch.Tensor(
            [[[-0.4831,  0.4514,  0.2072, -0.0963, -0.3155,  0.3777,  0.1536],
             [-0.4914,  0.4421,  0.1905, -0.1247, -0.3248,  0.3846,  0.1703],
             [-0.5011,  0.4363,  0.1793, -0.1462, -0.3347,  0.3919,  0.1790],
             [-0.5102,  0.4326,  0.1715, -0.1623, -0.3442,  0.3969,  0.1827]],

            [[-0.5211,  0.5055,  0.2877,  0.0200, -0.3148,  0.4124,  0.1030],
             [-0.5291,  0.4968,  0.2718, -0.0086, -0.3241,  0.4191,  0.1200],
             [-0.5384,  0.4913,  0.2610, -0.0304, -0.3340,  0.4263,  0.1288],
             [-0.5471,  0.4879,  0.2536, -0.0467, -0.3435,  0.4311,  0.1325]]])
        self.assertTensorAlmostEqual(hidden_target, hidden)
        self.assertTensorAlmostEqual(output_target, output)
        self.assertTensorAlmostEqual(att_vectors, att_vectors_target)
        # att_probs should be a distribution over the output vocabulary
        self.assertTensorAlmostEqual(att_probs.sum(2),
                                     torch.ones(batch_size, time_dim))

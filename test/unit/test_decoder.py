from test.unit.test_helpers import TensorTestCase

import torch
from torch.nn import GRU, LSTM

from joeynmt.decoders import RecurrentDecoder
from joeynmt.encoders import RecurrentEncoder


class TestRecurrentDecoder(TensorTestCase):

    def setUp(self):
        self.emb_size = 10
        self.num_layers = 3
        self.hidden_size = 6
        self.encoder_hidden_size = 3
        self.vocab_size = 5
        seed = 42
        torch.manual_seed(seed)

        bidi_encoder = RecurrentEncoder(
            emb_size=self.emb_size,
            num_layers=self.num_layers,
            hidden_size=self.encoder_hidden_size,
            bidirectional=True,
        )
        uni_encoder = RecurrentEncoder(
            emb_size=self.emb_size,
            num_layers=self.num_layers,
            hidden_size=self.encoder_hidden_size * 2,
            bidirectional=False,
        )
        self.encoders = [uni_encoder, bidi_encoder]

    def test_recurrent_decoder_size(self):
        # test all combinations of bridge, input_feeding, encoder directions
        for encoder in self.encoders:
            for init_hidden in ["bridge", "zero", "last"]:
                for input_feeding in [True, False]:
                    decoder = RecurrentDecoder(
                        hidden_size=self.hidden_size,
                        encoder=encoder,
                        attention="bahdanau",
                        emb_size=self.emb_size,
                        vocab_size=self.vocab_size,
                        num_layers=self.num_layers,
                        init_hidden=init_hidden,
                        input_feeding=input_feeding,
                    )
                    self.assertEqual(decoder.rnn.hidden_size, self.hidden_size)
                    self.assertEqual(decoder.att_vector_layer.out_features,
                                     self.hidden_size)
                    self.assertEqual(decoder.output_layer.out_features, self.vocab_size)
                    self.assertEqual(decoder.output_size, self.vocab_size)
                    self.assertEqual(decoder.rnn.bidirectional, False)

                    self.assertEqual(decoder.init_hidden_option, init_hidden)
                    if init_hidden == "bridge":
                        self.assertTrue(hasattr(decoder, "bridge_layer"))
                        self.assertEqual(decoder.bridge_layer.out_features,
                                         self.hidden_size)
                        self.assertEqual(
                            decoder.bridge_layer.in_features,
                            encoder.output_size,
                        )
                    else:
                        self.assertFalse(hasattr(decoder, "bridge_layer"))

                    if input_feeding:
                        self.assertEqual(
                            decoder.rnn_input_size,
                            self.emb_size + self.hidden_size,
                        )
                    else:
                        self.assertEqual(decoder.rnn_input_size, self.emb_size)

    def test_recurrent_decoder_type(self):
        valid_rnn_types = {"gru": GRU, "lstm": LSTM}
        for name, obj in valid_rnn_types.items():
            decoder = RecurrentDecoder(
                rnn_type=name,
                hidden_size=self.hidden_size,
                encoder=self.encoders[0],
                attention="bahdanau",
                emb_size=self.emb_size,
                vocab_size=self.vocab_size,
                num_layers=self.num_layers,
                init_hidden="zero",
                input_feeding=False,
            )
            self.assertEqual(type(decoder.rnn), obj)

    def test_recurrent_input_dropout(self):
        drop_prob = 0.5
        decoder = RecurrentDecoder(
            hidden_size=self.hidden_size,
            encoder=self.encoders[0],
            attention="bahdanau",
            emb_size=self.emb_size,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            init_hidden="zero",
            input_feeding=False,
            dropout=drop_prob,
            emb_dropout=drop_prob,
        )
        input_tensor = torch.Tensor([2, 3, 1, -1])
        decoder.train()
        dropped = decoder.emb_dropout(input=input_tensor)
        # eval switches off dropout
        decoder.eval()
        no_drop = decoder.emb_dropout(input=input_tensor)
        # when dropout is applied, remaining values are divided by drop_prob
        self.assertGreaterEqual((no_drop - (drop_prob * dropped)).abs().sum(), 0)

        drop_prob = 1.0
        decoder = RecurrentDecoder(
            hidden_size=self.hidden_size,
            encoder=self.encoders[0],
            attention="bahdanau",
            emb_size=self.emb_size,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            init_hidden="zero",
            input_feeding=False,
            dropout=drop_prob,
            emb_dropout=drop_prob,
        )
        all_dropped = decoder.emb_dropout(input=input_tensor)
        self.assertEqual(all_dropped.sum(), 0)
        decoder.eval()
        none_dropped = decoder.emb_dropout(input=input_tensor)
        self.assertTensorEqual(no_drop, none_dropped)
        self.assertTensorEqual((no_drop - all_dropped), no_drop)

    def test_recurrent_hidden_dropout(self):
        hidden_drop_prob = 0.5
        decoder = RecurrentDecoder(
            hidden_size=self.hidden_size,
            encoder=self.encoders[0],
            attention="bahdanau",
            emb_size=self.emb_size,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            init_hidden="zero",
            input_feeding=False,
            hidden_dropout=hidden_drop_prob,
        )
        input_tensor = torch.Tensor([2, 3, 1, -1])
        decoder.train()
        dropped = decoder.hidden_dropout(input=input_tensor)
        # eval switches off dropout
        decoder.eval()
        no_drop = decoder.hidden_dropout(input=input_tensor)
        # when dropout is applied, remaining values are divided by drop_prob
        self.assertGreaterEqual((no_drop - (hidden_drop_prob * dropped)).abs().sum(), 0)

        hidden_drop_prob = 1.0
        decoder = RecurrentDecoder(
            hidden_size=self.hidden_size,
            encoder=self.encoders[0],
            attention="bahdanau",
            emb_size=self.emb_size,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            init_hidden="zero",
            input_feeding=False,
            hidden_dropout=hidden_drop_prob,
        )
        all_dropped = decoder.hidden_dropout(input=input_tensor)
        self.assertEqual(all_dropped.sum(), 0)
        decoder.eval()
        none_dropped = decoder.hidden_dropout(input=input_tensor)
        self.assertTensorEqual(no_drop, none_dropped)
        self.assertTensorEqual((no_drop - all_dropped), no_drop)

    def test_recurrent_freeze(self):
        decoder = RecurrentDecoder(
            hidden_size=self.hidden_size,
            encoder=self.encoders[0],
            attention="bahdanau",
            emb_size=self.emb_size,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            init_hidden="zero",
            input_feeding=False,
            freeze=True,
        )
        for _, p in decoder.named_parameters():
            self.assertFalse(p.requires_grad)

    def test_recurrent_forward(self):
        time_dim = 4
        batch_size = 2
        # make sure the outputs match the targets
        decoder = RecurrentDecoder(
            hidden_size=self.hidden_size,
            encoder=self.encoders[0],
            attention="bahdanau",
            emb_size=self.emb_size,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            init_hidden="zero",
            input_feeding=False,
        )
        encoder_states = torch.rand(size=(batch_size, time_dim,
                                          self.encoders[0].output_size))
        trg_inputs = torch.ones(size=(batch_size, time_dim, self.emb_size))
        # no padding, no mask
        # x_length = torch.Tensor([time_dim]*batch_size).int()
        mask = torch.ones(size=(batch_size, 1, time_dim)).byte()
        output, hidden, att_probs, att_vectors = decoder(
            trg_inputs,
            encoder_hidden=encoder_states[:, -1, :],
            encoder_output=encoder_states,
            src_mask=mask,
            unroll_steps=time_dim,
            hidden=None,
            prev_att_vector=None,
        )
        self.assertEqual(output.shape,
                         torch.Size([batch_size, time_dim, self.vocab_size]))
        self.assertEqual(
            hidden.shape,
            torch.Size([batch_size, self.num_layers, self.hidden_size]),
        )
        self.assertEqual(att_probs.shape, torch.Size([batch_size, time_dim, time_dim]))
        self.assertEqual(
            att_vectors.shape,
            torch.Size([batch_size, time_dim, self.hidden_size]),
        )
        hidden_target = torch.Tensor([
            [
                [0.1814, 0.5468, -0.4717, -0.7580, 0.5834, -0.4018],
                [0.4649, 0.5484, -0.2702, 0.4545, 0.1983, 0.2771],
                [-0.1752, -0.4215, 0.1941, -0.3975, -0.2317, -0.5566],
            ],
            [
                [0.1814, 0.5468, -0.4717, -0.7580, 0.5834, -0.4018],
                [0.4649, 0.5484, -0.2702, 0.4545, 0.1983, 0.2771],
                [-0.1752, -0.4215, 0.1941, -0.3975, -0.2317, -0.5566],
            ],
        ])
        output_target = torch.Tensor([
            [
                [0.2702, -0.1988, -0.1985, -0.2998, -0.2564],
                [0.2719, -0.2075, -0.2017, -0.2988, -0.2595],
                [0.2720, -0.2143, -0.2084, -0.3024, -0.2537],
                [0.2714, -0.2183, -0.2135, -0.3061, -0.2468],
            ],
            [
                [0.2757, -0.1744, -0.1888, -0.3038, -0.2466],
                [0.2782, -0.1837, -0.1928, -0.3028, -0.2505],
                [0.2785, -0.1904, -0.1994, -0.3066, -0.2448],
                [0.2777, -0.1943, -0.2042, -0.3105, -0.2379],
            ],
        ])
        att_vectors_target = torch.Tensor([
            [
                [-0.6196, -0.0505, 0.4900, 0.6286, -0.5007, -0.3721],
                [-0.6389, -0.0337, 0.4998, 0.6458, -0.5052, -0.3579],
                [-0.6396, -0.0158, 0.5058, 0.6609, -0.5035, -0.3660],
                [-0.6348, -0.0017, 0.5090, 0.6719, -0.5013, -0.3771],
            ],
            [
                [-0.5697, -0.0887, 0.4515, 0.6128, -0.4713, -0.4068],
                [-0.5910, -0.0721, 0.4617, 0.6305, -0.4760, -0.3930],
                [-0.5918, -0.0544, 0.4680, 0.6461, -0.4741, -0.4008],
                [-0.5866, -0.0405, 0.4712, 0.6574, -0.4718, -0.4116],
            ],
        ])
        self.assertTensorAlmostEqual(hidden_target, hidden)
        self.assertTensorAlmostEqual(output_target, output)
        self.assertTensorAlmostEqual(att_vectors, att_vectors_target)
        # att_probs should be a distribution over the output vocabulary
        self.assertTensorAlmostEqual(att_probs.sum(2), torch.ones(batch_size, time_dim))

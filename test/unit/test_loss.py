from test.unit.test_helpers import TensorTestCase

import torch

from joeynmt.loss import XentLoss


class TestXentLoss(TensorTestCase):

    def setUp(self):
        seed = 42
        torch.manual_seed(seed)

    def test_label_smoothing(self):
        pad_index = 0
        smoothing = 0.4
        criterion = XentLoss(pad_index=pad_index, smoothing=smoothing)

        # batch x seq_len x vocab_size: 3 x 2 x 5
        predict = torch.FloatTensor([
            [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
            [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
            [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
        ])

        # batch x seq_len: 3 x 2
        targets = torch.LongTensor([[2, 1], [2, 0], [1, 0]])

        # test the smoothing function
        # pylint: disable=protected-access
        smoothed_targets = criterion._smooth_targets(targets=targets.view(-1),
                                                     vocab_size=predict.size(-1))
        # pylint: enable=protected-access
        self.assertTensorAlmostEqual(
            smoothed_targets,
            torch.Tensor([
                [0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
                [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
                [0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            ]),
        )
        assert torch.max(smoothed_targets) == 1 - smoothing

        # test the loss computation
        v = criterion(predict.log(), **{"trg": targets})
        self.assertTensorAlmostEqual(v, 2.1326)

    def test_no_label_smoothing(self):
        pad_index = 0
        smoothing = 0.0
        criterion = XentLoss(pad_index=pad_index, smoothing=smoothing)

        # batch x seq_len x vocab_size: 3 x 2 x 5
        predict = torch.FloatTensor([
            [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
            [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
            [[0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
        ])

        # batch x seq_len: 3 x 2
        targets = torch.LongTensor([[2, 1], [2, 0], [1, 0]])

        # test the smoothing function: should still be one-hot
        # pylint: disable=protected-access
        smoothed_targets = criterion._smooth_targets(targets=targets.view(-1),
                                                     vocab_size=predict.size(-1))
        # pylint: enable=protected-access

        assert torch.max(smoothed_targets) == 1
        assert torch.min(smoothed_targets) == 0

        self.assertTensorAlmostEqual(
            smoothed_targets,
            torch.Tensor([
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]),
        )

        v = criterion(predict.log(), **{"trg": targets})
        self.assertTensorAlmostEqual(v, 5.6268)

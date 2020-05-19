from test.unit.test_helpers import TensorTestCase

from joeynmt.metrics import token_accuracy


class TestMetrics(TensorTestCase):

    def test_token_acc_level_char(self):
        hyp = ["test"]
        ref = ["tezt"]
        level = "char"
        acc = token_accuracy(hyp, ref, level)
        self.assertEqual(acc, 75)

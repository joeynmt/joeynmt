import unittest

import torch


class TensorTestCase(unittest.TestCase):
    def assertTensorNotEqual(self, expected, actual):
        equal = torch.equal(expected, actual)
        if equal:
            self.fail("Tensors did match but weren't supposed to: expected {},"
                      " actual {}.".format(expected, actual))

    def assertTensorEqual(self, expected, actual):
        equal = torch.equal(expected, actual)
        if not equal:
            self.fail("Tensors didn't match but were supposed to {} vs"
                      " {}".format(expected, actual))

    def assertTensorAlmostEqual(self, expected, actual):
        diff = torch.all(
            torch.lt(torch.abs(torch.add(expected, -actual)), 1e-4))
        if not diff:
            self.fail("Tensors didn't match but were supposed to {} vs"
                      " {}".format(expected, actual))

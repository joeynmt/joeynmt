import unittest

import torch
from torch import Tensor


class TensorTestCase(unittest.TestCase):

    def assertTensorNotEqual(self, expected: Tensor, actual: Tensor):
        equal = torch.equal(expected, actual)
        if equal:
            self.fail(f"Tensors did match but weren't supposed to: "
                      f"expected {expected}, actual {actual}.")

    def assertTensorEqual(self, expected: Tensor, actual: Tensor):
        equal = torch.equal(expected, actual)
        if not equal:
            self.fail(f"Tensors didn't match but were supposed to "
                      f"{expected} vs {actual}")

    def assertTensorAlmostEqual(self, expected: Tensor, actual: Tensor):
        diff = torch.all(torch.lt(torch.abs(torch.add(expected, -actual)), 1e-4))
        if not diff:
            self.fail(f"Tensors didn't match but were supposed to "
                      f"{expected} vs {actual}")

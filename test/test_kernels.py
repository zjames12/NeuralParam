import unittest
from unittest import result
import torch
import numpy as np

from kernels.matern_nonstat_nugget import MaternNonstatNuggetKernel
from kernels.matern_nonstat_variance import MaternNonstatVarianceKernel


def constant_feature_extractor(x):
    """Neural network function that always returns 0.1"""
    return torch.full((x.shape[0], 1), 0.1)

def varying_feature_extractor(x):
    """Neural network function that returns [3,2,1]"""
    return torch.tensor([[3], [2], [1]], dtype=torch.float)



class TestMaternNonstatNuggetKernel(unittest.TestCase):
   
    def test_stationary_model(self):
        """Test that the kernel works with constant feature extractor returning 0.1"""
        a = torch.tensor([[1, 2],[1, 3], [2,2]], dtype=torch.float).view(3, 2)

        kernel = MaternNonstatNuggetKernel(lf=constant_feature_extractor, nu=0.5).initialize(lengthscale=1.0)
        kernel.eval()

        # Test that the feature extractor returns 0.1
        self.assertTrue(torch.all(constant_feature_extractor(a) == 0.1))

        # Test forward pass works
        result = kernel(a, a).to_dense()
        self.assertEqual(result.shape, (3, 3))

        # Diagonal elements should include the nugget (0.1) since x1[0] == x2[0]
        self.assertLess(result[0, 0].item() - 1.1, 1e-3)
        self.assertLess(result[1, 1].item() - 1.1, 1e-3)
        self.assertLess(result[2, 2].item() - 1.1, 1e-3)

class TestMaternNonstatVarianceKernel(unittest.TestCase):

    # def setUp(self):
    #     """Set up the kernel for stationary model tests"""
        

    def test_stationary_model(self):
        """Test that the kernel works with constant feature extractor returning 0.1"""
        a = torch.tensor([[1, 2],[1, 3], [2,2]], dtype=torch.float).view(3, 2)

        self.kernel = MaternNonstatVarianceKernel(lf=constant_feature_extractor, nu=0.5)#.initialize(lengthscale=1.0)
        self.kernel.eval()

        # Test that the feature extractor returns 0.1
        self.assertTrue(torch.all(constant_feature_extractor(a) == 0.1))

        # Test forward pass works
        result = self.kernel(a, a).to_dense()
        self.assertEqual(result.shape, (3, 3))

        # Test if result equals the expected matrix
        expected = torch.tensor([
            [0.010000000, 0.003678794, 0.003678794],
            [0.003678794, 0.010000000, 0.002431167],
            [0.003678794, 0.002431167, 0.010000000]
        ])
        self.assertTrue(torch.allclose(result, expected, rtol=1e-6, atol=1e-5))

    def test_non_stationary_model(self):
        """Test that the kernel works with varying feature extractor"""
        a = torch.tensor([[1, 2],[1, 3], [2,2]], dtype=torch.float).view(3, 2)

        self.kernel = MaternNonstatVarianceKernel(lf=varying_feature_extractor, nu=0.5)#.initialize(lengthscale=1.0)
        self.kernel.eval()

        # Test that the feature extractor returns 0.1
        self.assertTrue(torch.all(varying_feature_extractor(a) == torch.tensor([[3], [2], [1]], dtype=torch.float)))

        # Test forward pass works
        result = self.kernel(a, a).to_dense()
        self.assertEqual(result.shape, (3, 3))

        # Test if result equals the expected matrix
        expected = torch.tensor([
            [9.000000, 2.2072766, 1.1036383],
            [2.207277, 4.0000000, 0.4862335],
            [1.103638, 0.4862335, 1.0000000]
        ])

        self.assertTrue(torch.allclose(result, expected, rtol=1e-6, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
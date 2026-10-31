import unittest
from unittest import result
import torch
import numpy as np

class TestSphericalCoordinates(unittest.TestCase):

    def test_spherical_coordinates(self):
        """Test that the spherical_coordinates function correctly converts Cartesian coordinates to spherical coordinates"""
        from functions.spherical_distances import spherical_coordinates

        # Test input: (longitude, latitude)
        input_coords = torch.tensor([[0, 0], [90, 0], [180, 0], [270, 0], [0, 90], [0, -90]], dtype=torch.float)
        expected_output = torch.tensor([
            [1.0, 0.0, 0.0],   # (0, 0) -> (x=0, y=0, z=1)
            [0.0, 1.0, 0.0],   # (90, 0) -> (x=1, y=0, z=0)
            [-1.0, 0.0, 0.0], # (180, 0) -> (x=0, y~0, z=-1)
            [0.0, -1.0, 0.0], # (270, 0) -> (x~0, y~0, z~0)
            [0.0, 0.0, -1.0], # (0, 90) -> (x~0, y~0, z~1)
            [0.0, 0.0, 1.0] # (0, -90) -> (x~0, y~0, z~-1)
        ], dtype=torch.float)

        output = spherical_coordinates(input_coords)

        self.assertTrue(torch.allclose(output, expected_output, atol=1e-5))
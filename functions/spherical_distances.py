import torch
import numpy as np


def spherical_coordinates(x : torch.Tensor) -> torch.Tensor:
    """
    Convert Cartesian coordinates to spherical coordinates.

    Args:
        x: Tensor of shape (n, 2) representing the set of points (longitude, latitude)

    Returns:
        Tensor of shape (n, 3) containing the spherical coordinates (x, y, z)
    """
    # Compute the radius
    lonrad = 2 * torch.pi * x[:, 0] / 360
    latrad = 2 * torch.pi * (x[:, 1] + 90) / 360
    xyz = torch.zeros(x.shape[0], 3, device=x.device)
    xyz[:, 0] = torch.sin(latrad) * torch.cos(lonrad)  # x coordinate
    xyz[:, 1] = torch.sin(latrad) * torch.sin(lonrad)  # y coordinate
    xyz[:, 2] = torch.cos(latrad)  # z coordinate
    return xyz


def inverse_spherical_coordinates(xyz: torch.Tensor) -> torch.Tensor:
    """
    Convert spherical coordinates back to Cartesian coordinates (longitude, latitude).

    Args:
        xyz: Tensor of shape (n, 3) representing spherical coordinates (x, y, z)

    Returns:
        Tensor of shape (n, 2) containing the coordinates (longitude, latitude) in degrees
    """
    # Extract latitude and longitude from spherical coordinates
    latrad = torch.arccos(xyz[:, 2])  # z = cos(latrad)
    lonrad = torch.atan2(xyz[:, 1], xyz[:, 0])  # atan2(y, x) where y = sin(latrad)*sin(lonrad), x = sin(latrad)*cos(lonrad)

    # Convert back to degrees
    lon = lonrad * 360 / (2 * torch.pi)
    lat = latrad * 360 / (2 * torch.pi) - 90

    # Stack longitude and latitude
    result = torch.zeros(xyz.shape[0], 2, device=xyz.device)
    result[:, 0] = lon
    result[:, 1] = lat
    return result


def spherical_coordinates_numpy(x : np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates to spherical coordinates using numpy.

    Args:
        x: NumPy array of shape (n, 2) representing the set of points (longitude, latitude)

    Returns:
        NumPy array of shape (n, 3) containing the spherical coordinates (x, y, z)
    """
    # Compute the radius
    lonrad = 2 * np.pi * x[:, 0] / 360
    latrad = 2 * np.pi * (x[:, 1] + 90) / 360
    xyz = np.zeros((x.shape[0], 3))
    xyz[:, 0] = np.sin(latrad) * np.cos(lonrad)  # x coordinate
    xyz[:, 1] = np.sin(latrad) * np.sin(lonrad)  # y coordinate
    xyz[:, 2] = np.cos(latrad)  # z coordinate
    return xyz


# Gaussian basis function network for neural parametrization

import torch
import torch.nn as nn

from functions.spherical_distances import spherical_coordinates

import numpy as np # <- Delete after testing


class BasisFunction(nn.Module):
    """
    Gaussian basis function network with q×p basis functions arranged in a 2D grid.

    Args:
        q: Number of basis functions along x-axis
        p: Number of basis functions along y-axis (total basis functions = q×p)
        x_bounds: Tuple of (x_min, x_max) for the x spatial domain
        y_bounds: Tuple of (y_min, y_max) for the y spatial domain
        std: Standard deviation for the Gaussian basis functions (default: 1.0)

    Example:
        # Create a 4x3 = 12 basis function network with different x and y domains
        >>> import torch
        >>> basis_net = BasisFunction(q=4, p=3, x_bounds=(-2, 2), y_bounds=(-1, 3), std=0.5)
        >>>
        >>> # Forward pass with batch of 2D points
        >>> x = torch.tensor([[0.0, 0.0], [1.0, -0.5], [-1.5, 2.5]])  # (3, 2)
        >>> output = basis_net(x)  # (3, 1) - neural network output
        >>> print(f"Network output shape: {output.shape}")
        >>>
        >>> # Get the grid centers (useful for visualization)
        >>> centers = basis_net.get_centers()  # (12, 2)
        >>> print(f"Centers shape: {centers.shape}")
        >>> print(f"First few centers: {centers[:3]}")
        >>>
        >>> # Get basis function values without linear combination
        >>> basis_values = basis_net.get_basis_values(x)  # (3, 12)
        >>> print(f"Basis values shape: {basis_values.shape}")
        >>> print(f"Sum of basis values for first point: {basis_values[0].sum()}")
        >>>
        >>> # Square grid example (same as before)
        >>> square_net = BasisFunction(q=4, p=4, x_bounds=(-2, 2), y_bounds=(-2, 2), std=0.5)
        >>>
        >>> # Training example
        >>> optimizer = torch.optim.Adam(basis_net.parameters(), lr=0.01)
        >>> target = torch.randn(3, 1)  # some target values
        >>>
        >>> # Forward pass
        >>> pred = basis_net(x)
        >>> loss = torch.nn.functional.mse_loss(pred, target)
        >>>
        >>> # Backward pass
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """
    def __init__(self, q, p, x_bounds, y_bounds, std=1.0, sphere = True, sigmoid = False):
        super(BasisFunction, self).__init__()

        self.q = q
        self.p = p
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.std = std
        self.sphere = sphere

        # Create a q x p grid of centers in 2D space
        x = torch.linspace(x_bounds[0], x_bounds[1], q)
        y = torch.linspace(y_bounds[0], y_bounds[1], p)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        centers_x = xx.flatten()
        centers_y = yy.flatten()
        centers = torch.stack([centers_x, centers_y], dim=1)  # Shape: (q×p, 2)
        self.n_basis = q * p
        # np.savetxt("knots.csv", centers.numpy(), delimiter=",")
        
        if self.sphere:
            centers = self.fibonacci_sphere(2500)  # Use Fibonacci sphere for more uniform coverage
            centers = spherical_coordinates(centers)
            self.n_basis = 2500

        # Register centers as buffer so they move with device but aren't parameters
        self.register_buffer('centers', centers)

        # Linear layer for combining basis functions (neural network)
        self.linear = nn.Linear(self.n_basis, 2)
        if sigmoid:
            self.softplus = nn.Sigmoid()
        else:
            self.softplus = nn.Softplus()

    def gaussian_basis(self, x):
        """
        Calculate Gaussian basis function values.

        Args:
            x: Input tensor of shape (batch_size, 2)

        Returns:
            Basis function values of shape (batch_size, q×p)
        """
        # Compute squared distances from each input to each center
        # x shape: (batch_size, 2), centers shape: (q×p, 2)
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, 2)
        centers_expanded = self.centers.unsqueeze(0)  # (1, q×p, 2)

        # Squared Euclidean distance
        squared_dist = torch.sum((x_expanded - centers_expanded) ** 2, dim=-1)

        # Gaussian basis functions: exp(-||x - c||^2 / (2 * std^2))
        basis_values = torch.exp(-squared_dist / (2 * self.std ** 2))

        return basis_values

    def forward(self, x):
        """
        Forward pass through basis functions and linear combination.

        Args:
            x: Input tensor of shape (batch_size, 2)

        Returns:
            Output tensor of shape (batch_size, 1) - positive values due to Softplus
        """
        # Calculate basis function values
        basis_values = self.gaussian_basis(x)

        # Linear combination of basis functions (neural network)
        linear_output = self.linear(basis_values)

        # Apply Softplus for positive output
        output = self.softplus(linear_output)

        return output

    def get_centers(self):
        """
        Get the centers of the Gaussian basis functions.

        Returns:
            Tensor of shape (q×p, 2) containing the centers
        """
        return self.centers

    def get_basis_values(self, x):
        """
        Get the basis function values without the linear combination.
        Useful for visualization or analysis.

        Args:
            x: Input tensor of shape (batch_size, 2)

        Returns:
            Basis function values of shape (batch_size, q×p)
        """
        return self.gaussian_basis(x)
    
    # Create a roughly uniform grid
    def fibonacci_sphere(self, samples=100):
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # Golden angle
        
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)
            
            theta = phi * i  # Golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            # Convert to lat/lon
            lat = np.arcsin(y) * 180 / np.pi
            lon = np.arctan2(z, x) * 180 / np.pi
            points.append((lon, lat))
        
        return torch.tensor(points)


class LargeBasisFunction(nn.Module):
    """
    Larger basis function network with multiple hidden layers after the basis functions.

    Args:
        q: Number of basis functions along x-axis
        p: Number of basis functions along y-axis (total basis functions = q×p)
        x_bounds: Tuple of (x_min, x_max) for the x spatial domain
        y_bounds: Tuple of (y_min, y_max) for the y spatial domain
        std: Standard deviation for the Gaussian basis functions (default: 1.0)
        hidden_dims: List of hidden layer dimensions (default: [25, 10])
        dropout_p: Dropout probability (default: 0.0)
    """
    def __init__(self, q, p, x_bounds, y_bounds, std=1.0, hidden_dims=[25, 10], dropout_p=0.0):
        super(LargeBasisFunction, self).__init__()

        self.q = q
        self.p = p
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.std = std

        # Create a q x p grid of centers in 2D space
        x = torch.linspace(x_bounds[0], x_bounds[1], q)
        y = torch.linspace(y_bounds[0], y_bounds[1], p)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        centers_x = xx.flatten()
        centers_y = yy.flatten()
        centers = torch.stack([centers_x, centers_y], dim=1)  # Shape: (q×p, 2)
        self.n_basis = q * p

        self.register_buffer('centers', centers)

        # Build the neural network
        layers = []
        in_dim = self.n_basis

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            in_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Softplus())

        self.network = nn.Sequential(*layers)

    def gaussian_basis(self, x):
        """
        Calculate Gaussian basis function values.

        Args:
            x: Input tensor of shape (batch_size, 2)

        Returns:
            Basis function values of shape (batch_size, q×p)
        """
        x_expanded = x.unsqueeze(1)
        centers_expanded = self.centers.unsqueeze(0)
        squared_dist = torch.sum((x_expanded - centers_expanded) ** 2, dim=-1)
        basis_values = torch.exp(-squared_dist / (2 * self.std ** 2))
        return basis_values

    def forward(self, x):
        """
        Forward pass through basis functions and deep neural network.

        Args:
            x: Input tensor of shape (batch_size, 2)

        Returns:
            Output tensor of shape (batch_size, 1) - positive values due to Softplus
        """
        basis_values = self.gaussian_basis(x)
        output = self.network(basis_values)
        return output

    def get_centers(self):
        """
        Get the centers of the Gaussian basis functions.

        Returns:
            Tensor of shape (q×p, 2) containing the centers
        """
        return self.centers

    def get_basis_values(self, x):
        """
        Get the basis function values without the neural network.

        Args:
            x: Input tensor of shape (batch_size, 2)

        Returns:
            Basis function values of shape (batch_size, q×p)
        """
        return self.gaussian_basis(x)
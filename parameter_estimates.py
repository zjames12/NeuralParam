# Parameter estimation and prediction using saved feature extractor weights

from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from basis_function import BasisFunction
from feature_extractor import LargeFeatureExtractor
from functions.spherical_distances import spherical_coordinates_numpy

np.random.seed(0)
torch.manual_seed(0)

def load_feature_extractor(weights_path, n_dim, device='cpu'):
    """
    Load a trained feature extractor from saved weights.

    Args:
        weights_path: Path to the saved weights file (.pth)
        n_dim: Input dimensionality used during training
        device: Device to load the model on ('cpu' or 'cuda')s

    Returns:
        Loaded feature extractor model
    """
    # Initialize the feature extractor
    feature_extractor = LargeFeatureExtractor(n_dim, sigmoid=True)
    feature_extractor = BasisFunction(q=50, p=50, x_bounds=(-96, -91), y_bounds=(34,37.5), std=0.4, sphere=True, sigmoid = True)


    # feature_extractor = BasisFunction(q=12, p=8, x_bounds=(-125, -65), y_bounds=(24,53), std=6)
    # feature_extractor = BasisFunction(q=40, p=29, x_bounds=(-125, -65), y_bounds=(24,53), std=2, sphere=False, sigmoid=True)
    # feature_extractor = BasisFunction(q=70, p=35, x_bounds=(-180, 180), y_bounds=(-90,90), std=50/360)
    # feature_extractor = BasisFunction(q=40, p=40, x_bounds=(0, 225), y_bounds=(0,150), std=4)
    
    # feature_extractor = BasisFunction(q=50, p=50, x_bounds=(-96, -91), y_bounds=(34,37.5), std=0.4, sphere=False, sigmoid = True)
    # feature_extractor = BasisFunction(q=40, p=29, x_bounds=(-125, -65), y_bounds=(24,53), std=2, sphere=False, sigmoid = False)
    
    # feature_extractor = BasisFunction(q=84, p=42, x_bounds=(-180, 180), y_bounds=(-90,90), std=40/360, sphere=True, sigmoid=True)

    # Load the saved state dictionary
    weights_path = os.path.expanduser(weights_path)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)

    # trained_model.pth is a full GP state dict; extract the feature extractor sub-dict
    prefix = 'covar_module.base_kernel.feature_extractor.'
    # prefix = 'covar_module.feature_extractor.'
    if any(k.startswith(prefix) for k in state_dict):
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    feature_extractor.load_state_dict(state_dict)

    # Move to specified device
    feature_extractor = feature_extractor.to(device)

    # Set to evaluation mode
    feature_extractor.eval()

    print(f"Feature extractor loaded from: {weights_path}")
    print(f"Architecture: Deep")
    print(f"Input dimensions: {n_dim}")
    print(f"Device: {device}")

    return feature_extractor


def make_predictions(feature_extractor, X_input, device='cpu'):
    """
    Make predictions using the loaded feature extractor.

    Args:
        feature_extractor: Loaded feature extractor model
        X_input: Input data (numpy array or torch tensor)
        device: Device to run predictions on

    Returns:
        Predictions as numpy array
    """
    # Convert input to torch tensor if needed
    if isinstance(X_input, np.ndarray):
        X_input = torch.from_numpy(X_input).float()

    # Move to device
    X_input = X_input.to(device)

    # Make predictions
    feature_extractor.eval()
    with torch.no_grad():
        predictions = feature_extractor(X_input)

    # Convert back to numpy
    predictions = predictions.cpu().numpy()

    return predictions


def analyze_parameters(feature_extractor):
    """
    Analyze the learned parameters of the feature extractor.

    Args:
        feature_extractor: Loaded feature extractor model

    Returns:
        Dictionary containing parameter analysis
    """
    analysis = {}

    print("=== Feature Extractor Parameter Analysis ===")

    for name, param in feature_extractor.named_parameters():
        param_data = param.data.cpu().numpy()

        analysis[name] = {
            'shape': param_data.shape,
            'mean': np.mean(param_data),
            'std': np.std(param_data),
            'min': np.min(param_data),
            'max': np.max(param_data)
        }

        print(f"\n{name}:")
        print(f"  Shape: {param_data.shape}")
        print(f"  Mean: {np.mean(param_data):.6f}")
        print(f"  Std:  {np.std(param_data):.6f}")
        print(f"  Range: [{np.min(param_data):.6f}, {np.max(param_data):.6f}]")

    return analysis


# Example usage
if __name__ == "__main__":
    # Example: Load a feature extractor and make predictions

    # These parameters should match your training configuration
    n_dim = 3  # Replace with your actual input dimensions
    # weights_path = "runs/experiment_1782397925/feature_extractor_weights.pth"  # Replace with actual path
    weights_path = "~/Documents/nonstat/runs/experiment_1783025947/trained_model.pth"  # Replace with actual path

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        # Load the feature extractor
        feature_extractor = load_feature_extractor(weights_path, n_dim, device)
        # feature_extractor = LargeFeatureExtractor(n_dim)

        # Analyze parameters
        # parameter_analysis = analyze_parameters(feature_extractor)

        # Create some example input data for predictions
        # Replace this with your actual input data
        # df = pd.read_csv("~/nonstat/heaton.csv", header = 0)
        # df = pd.read_csv("~/Documents/nonstat/ncar.csv", header = 0)
        df = pd.read_csv("~/Documents/nonstat/braceu.csv", header = 0)


        # # Convert last two columns to spherical coordinates (3 columns)
        latlon = df.iloc[:,1:3].values
        spherical_coords = spherical_coordinates_numpy(df.iloc[:,1:3].values)
        df = df.drop(df.columns[1:3], axis=1)  # Remove last two columns
        spherical_df = pd.DataFrame(spherical_coords, columns=['x', 'y', 'z'])
        df = pd.concat([df, spherical_df], axis=1)

        # scaler = StandardScaler()
        # df.iloc[:,1:] = scaler.fit_transform(df.iloc[:,1:]).copy()


        X_example = torch.as_tensor(df.iloc[:,1:].values).float()
        # X_example = np.random.randn(10, n_dim)  # 10 samples with n_dim features

        # Make predictions
        predictions = make_predictions(feature_extractor, X_example, device)

        # Save predictions with X_example as CSV
        combined_data = np.column_stack((predictions, latlon))
        pd.DataFrame(combined_data).to_csv("~/Documents/nonstat/braceu_nu_d.csv", index=False, header=False)
        print("Predictions saved to ~/Documents/nonstat/braceu_nu_d.csv")

        print(f"\n=== Predictions ===")
        print(f"Input shape: {X_example.shape}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Sample predictions:")
        # 
        print(pd.DataFrame(combined_data).describe())

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please update the weights_path variable with the correct path to your saved weights.")
    except Exception as e:
        print(f"An error occurred: {e}")
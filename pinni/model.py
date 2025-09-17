"""PINNI Neural Network Model

Implements the Physics-Informed Neural-Network Integration model
with 5 hidden layers, 256 neurons each, and Tanh activation functions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class PINNIModel(nn.Module):
    """Physics-Informed Neural-Network Integration Model
    
    A shallow multilayer perceptron designed to approximate the integrand function
    g(x; ε) in mechanical constitutive relations. The model uses Tanh activation
    functions to ensure smoothness and integrability.
    
    Architecture:
    - Input: concatenation of integration variable x and strain tensor features
    - 5 hidden layers with 256 neurons each
    - Tanh activation functions
    - Single output: predicted integrand value
    
    Args:
        input_dim (int): Dimension of input features (x + strain features)
        hidden_dim (int): Number of neurons in each hidden layer (default: 256)
        num_layers (int): Number of hidden layers (default: 5)
        dropout_rate (float): Dropout rate for regularization (default: 0.1)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 5,
        dropout_rate: float = 0.1
    ):
        super(PINNIModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # Build the network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
                             Contains concatenated [integration_var, strain_features]
        
        Returns:
            torch.Tensor: Predicted integrand values of shape (batch_size, 1)
        """
        return self.network(x)
    
    def predict_integrand(
        self,
        integration_points: torch.Tensor,
        strain_features: torch.Tensor
    ) -> torch.Tensor:
        """Predict integrand values for given integration points and strain state
        
        Args:
            integration_points (torch.Tensor): Integration variable values (N,)
            strain_features (torch.Tensor): Strain tensor features (strain_dim,)
        
        Returns:
            torch.Tensor: Predicted integrand values (N,)
        """
        # Ensure inputs are tensors
        if not isinstance(integration_points, torch.Tensor):
            integration_points = torch.tensor(integration_points, dtype=torch.float32)
        if not isinstance(strain_features, torch.Tensor):
            strain_features = torch.tensor(strain_features, dtype=torch.float32)
        
        # Reshape integration points to column vector
        if integration_points.dim() == 1:
            integration_points = integration_points.unsqueeze(1)
        
        # Expand strain features to match batch size
        batch_size = integration_points.size(0)
        strain_expanded = strain_features.unsqueeze(0).expand(batch_size, -1)
        
        # Concatenate inputs
        inputs = torch.cat([integration_points, strain_expanded], dim=1)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(inputs)
        
        return outputs.squeeze(1)
    
    def get_model_info(self) -> dict:
        """Get model architecture information
        
        Returns:
            dict: Model information including parameters count and architecture
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'activation_function': 'Tanh',
            'architecture': 'Shallow MLP for integrand approximation'
        }
    
    def save_model(self, filepath: str):
        """Save model state dict and configuration
        
        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout_rate': self.dropout_rate
            }
        }
        torch.save(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str, device: Optional[str] = None) -> 'PINNIModel':
        """Load model from saved state
        
        Args:
            filepath (str): Path to the saved model
            device (str, optional): Device to load the model on
        
        Returns:
            PINNIModel: Loaded model instance
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model_data = torch.load(filepath, map_location=device)
        config = model_data['config']
        
        # Create model instance
        model = cls(**config)
        model.load_state_dict(model_data['state_dict'])
        model.to(device)
        
        return model


class StrainFeatureExtractor:
    """Extract features from strain tensor for neural network input
    
    Converts strain tensor into meaningful features that can be used
    as input to the PINNI model along with the integration variable.
    """
    
    @staticmethod
    def extract_features(strain_tensor: np.ndarray) -> np.ndarray:
        """Extract features from strain tensor
        
        Args:
            strain_tensor (np.ndarray): Strain tensor (can be 1D, 2D, or 3D)
        
        Returns:
            np.ndarray: Extracted features including invariants and components
        """
        # Ensure strain tensor is numpy array
        strain = np.asarray(strain_tensor)
        
        # For simplicity, we'll use the strain components directly
        # In practice, you might want to compute strain invariants
        if strain.ndim == 1:
            # 1D case: direct components
            features = strain.flatten()
        elif strain.ndim == 2:
            # 2D case: matrix components
            features = strain.flatten()
        else:
            # Higher dimensional case
            features = strain.flatten()
        
        # Add strain invariants for better physical representation
        if len(features) >= 3:
            # First invariant (trace)
            I1 = np.sum(features[:3]) if len(features) >= 3 else features[0]
            
            # Second invariant (simplified)
            I2 = np.sum(features**2)
            
            # Add invariants to features
            features = np.concatenate([features, [I1, I2]])
        
        return features
    
    @staticmethod
    def get_feature_dim(strain_tensor_shape: Tuple[int, ...]) -> int:
        """Get the dimension of extracted features
        
        Args:
            strain_tensor_shape (Tuple[int, ...]): Shape of the strain tensor
        
        Returns:
            int: Dimension of extracted features
        """
        # Base features from tensor components
        base_dim = np.prod(strain_tensor_shape)
        
        # Additional invariant features
        invariant_dim = 2 if base_dim >= 3 else 0
        
        return base_dim + invariant_dim
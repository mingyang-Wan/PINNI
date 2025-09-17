"""Physics-Informed Loss Functions for PINNI

Implements the physics-informed loss function that combines:
1. Function fitting loss (L_func): MSE between predicted and true integrand values
2. Physics constraint loss (L_physic): Error in integral approximation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict


class PINNILoss(nn.Module):
    """Physics-Informed Neural Network Integration Loss Function
    
    Combines pointwise function fitting loss with physics-based integral constraint loss.
    The total loss is: L_total = L_func + λ * L_physic
    
    Args:
        lambda_physics (float): Weight for physics constraint loss (default: 1.0)
        integration_method (str): Method for numerical integration ('trapezoidal' or 'simpson')
        reduction (str): Reduction method for loss ('mean', 'sum', or 'none')
    """
    
    def __init__(
        self,
        lambda_physics: float = 1.0,
        integration_method: str = 'trapezoidal',
        reduction: str = 'mean'
    ):
        super(PINNILoss, self).__init__()
        
        self.lambda_physics = lambda_physics
        self.integration_method = integration_method
        self.reduction = reduction
        
        # MSE loss for function fitting
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        model: nn.Module,
        x_func: torch.Tensor,
        y_func: torch.Tensor,
        x_physics: torch.Tensor,
        strain_features: torch.Tensor,
        I_true: torch.Tensor,
        integration_bounds: Tuple[float, float]
    ) -> Dict[str, torch.Tensor]:
        """Compute the total physics-informed loss
        
        Args:
            model (nn.Module): PINNI model
            x_func (torch.Tensor): Input points for function fitting (M, input_dim)
            y_func (torch.Tensor): True integrand values for function fitting (M,)
            x_physics (torch.Tensor): Integration points for physics constraint (N,)
            strain_features (torch.Tensor): Strain tensor features (strain_dim,)
            I_true (torch.Tensor): True integral value (scalar or batch)
            integration_bounds (Tuple[float, float]): Integration domain [a, b]
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing individual and total losses
        """
        # Function fitting loss
        y_pred_func = model(x_func)
        if y_pred_func.dim() > 1:
            y_pred_func = y_pred_func.squeeze(-1)
        
        L_func = self.mse_loss(y_pred_func, y_func)
        
        # Physics constraint loss
        L_physic = self._compute_physics_loss(
            model, x_physics, strain_features, I_true, integration_bounds
        )
        
        # Total loss
        L_total = L_func + self.lambda_physics * L_physic
        
        return {
            'total_loss': L_total,
            'function_loss': L_func,
            'physics_loss': L_physic,
            'lambda_physics': torch.tensor(self.lambda_physics)
        }
    
    def _compute_physics_loss(
        self,
        model: nn.Module,
        x_physics: torch.Tensor,
        strain_features: torch.Tensor,
        I_true: torch.Tensor,
        integration_bounds: Tuple[float, float]
    ) -> torch.Tensor:
        """Compute physics constraint loss using numerical integration
        
        Args:
            model (nn.Module): PINNI model
            x_physics (torch.Tensor): Integration points (N,)
            strain_features (torch.Tensor): Strain tensor features
            I_true (torch.Tensor): True integral value
            integration_bounds (Tuple[float, float]): Integration domain [a, b]
        
        Returns:
            torch.Tensor: Physics constraint loss
        """
        # Prepare inputs for model
        batch_size = x_physics.size(0)
        
        # Expand strain features to match batch size
        if strain_features.dim() == 1:
            strain_expanded = strain_features.unsqueeze(0).expand(batch_size, -1)
        else:
            strain_expanded = strain_features
        
        # Reshape x_physics if needed
        if x_physics.dim() == 1:
            x_physics = x_physics.unsqueeze(1)
        
        # Concatenate inputs
        model_inputs = torch.cat([x_physics, strain_expanded], dim=1)
        
        # Get model predictions
        y_pred_physics = model(model_inputs)
        if y_pred_physics.dim() > 1:
            y_pred_physics = y_pred_physics.squeeze(-1)
        
        # Compute numerical integral
        I_pred = self._numerical_integration(
            y_pred_physics, x_physics.squeeze(-1), integration_bounds
        )
        
        # Compute physics loss
        if self.reduction == 'mean':
            L_physic = torch.mean((I_pred - I_true) ** 2)
        elif self.reduction == 'sum':
            L_physic = torch.sum((I_pred - I_true) ** 2)
        else:
            L_physic = (I_pred - I_true) ** 2
        
        return L_physic
    
    def _numerical_integration(
        self,
        y_values: torch.Tensor,
        x_points: torch.Tensor,
        integration_bounds: Tuple[float, float]
    ) -> torch.Tensor:
        """Perform numerical integration using specified method
        
        Args:
            y_values (torch.Tensor): Function values at integration points (N,)
            x_points (torch.Tensor): Integration points (N,)
            integration_bounds (Tuple[float, float]): Integration domain [a, b]
        
        Returns:
            torch.Tensor: Approximated integral value
        """
        if self.integration_method == 'trapezoidal':
            return self._trapezoidal_rule(y_values, x_points)
        elif self.integration_method == 'simpson':
            return self._simpson_rule(y_values, x_points)
        else:
            raise ValueError(f"Unknown integration method: {self.integration_method}")
    
    def _trapezoidal_rule(
        self,
        y_values: torch.Tensor,
        x_points: torch.Tensor
    ) -> torch.Tensor:
        """Trapezoidal rule for numerical integration
        
        Implements: ∫f(x)dx ≈ Σ[i=0 to N-2] (f(x[i+1]) + f(x[i])) * (x[i+1] - x[i]) / 2
        
        Args:
            y_values (torch.Tensor): Function values (N,)
            x_points (torch.Tensor): Integration points (N,)
        
        Returns:
            torch.Tensor: Approximated integral
        """
        # Sort points if not already sorted
        sorted_indices = torch.argsort(x_points)
        x_sorted = x_points[sorted_indices]
        y_sorted = y_values[sorted_indices]
        
        # Compute differences
        dx = x_sorted[1:] - x_sorted[:-1]
        
        # Trapezoidal rule
        integral = torch.sum(0.5 * (y_sorted[1:] + y_sorted[:-1]) * dx)
        
        return integral
    
    def _simpson_rule(
        self,
        y_values: torch.Tensor,
        x_points: torch.Tensor
    ) -> torch.Tensor:
        """Simpson's rule for numerical integration
        
        Args:
            y_values (torch.Tensor): Function values (N,)
            x_points (torch.Tensor): Integration points (N,)
        
        Returns:
            torch.Tensor: Approximated integral
        """
        # Sort points
        sorted_indices = torch.argsort(x_points)
        x_sorted = x_points[sorted_indices]
        y_sorted = y_values[sorted_indices]
        
        n = len(x_sorted)
        if n < 3:
            # Fall back to trapezoidal rule for insufficient points
            return self._trapezoidal_rule(y_values, x_points)
        
        # Simpson's rule requires even number of intervals
        if n % 2 == 0:
            # Use composite Simpson's rule with trapezoidal for last interval
            integral_simpson = self._composite_simpson(y_sorted[:-1], x_sorted[:-1])
            # Add last interval using trapezoidal rule
            integral_trap = 0.5 * (y_sorted[-1] + y_sorted[-2]) * (x_sorted[-1] - x_sorted[-2])
            integral = integral_simpson + integral_trap
        else:
            integral = self._composite_simpson(y_sorted, x_sorted)
        
        return integral
    
    def _composite_simpson(
        self,
        y_values: torch.Tensor,
        x_points: torch.Tensor
    ) -> torch.Tensor:
        """Composite Simpson's rule
        
        Args:
            y_values (torch.Tensor): Function values (odd number of points)
            x_points (torch.Tensor): Integration points (odd number of points)
        
        Returns:
            torch.Tensor: Approximated integral
        """
        n = len(x_points)
        h = (x_points[-1] - x_points[0]) / (n - 1)
        
        # Simpson's rule: ∫f(x)dx ≈ h/3 * [f(x0) + 4*Σf(x_odd) + 2*Σf(x_even) + f(xn)]
        integral = y_values[0] + y_values[-1]
        
        # Add odd indices with weight 4
        for i in range(1, n-1, 2):
            integral += 4 * y_values[i]
        
        # Add even indices with weight 2
        for i in range(2, n-1, 2):
            integral += 2 * y_values[i]
        
        integral *= h / 3
        
        return integral


class AdaptivePINNILoss(PINNILoss):
    """Adaptive Physics-Informed Loss with dynamic lambda adjustment
    
    Automatically adjusts the physics constraint weight based on the
    relative magnitudes of function and physics losses.
    """
    
    def __init__(
        self,
        lambda_physics: float = 1.0,
        adaptation_rate: float = 0.1,
        min_lambda: float = 0.01,
        max_lambda: float = 100.0,
        **kwargs
    ):
        super().__init__(lambda_physics=lambda_physics, **kwargs)
        
        self.adaptation_rate = adaptation_rate
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.loss_history = []
    
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive lambda adjustment"""
        # Compute losses
        losses = super().forward(*args, **kwargs)
        
        # Update lambda based on loss ratio
        self._update_lambda(losses['function_loss'], losses['physics_loss'])
        
        # Recompute total loss with updated lambda
        losses['total_loss'] = (
            losses['function_loss'] + self.lambda_physics * losses['physics_loss']
        )
        losses['lambda_physics'] = torch.tensor(self.lambda_physics)
        
        return losses
    
    def _update_lambda(self, L_func: torch.Tensor, L_physic: torch.Tensor):
        """Update lambda based on loss magnitudes"""
        # Store current losses
        self.loss_history.append({
            'function_loss': L_func.item(),
            'physics_loss': L_physic.item()
        })
        
        # Keep only recent history
        if len(self.loss_history) > 10:
            self.loss_history.pop(0)
        
        # Adjust lambda if we have enough history
        if len(self.loss_history) >= 3:
            # Compute average loss ratio
            func_losses = [h['function_loss'] for h in self.loss_history[-3:]]
            phys_losses = [h['physics_loss'] for h in self.loss_history[-3:]]
            
            avg_func = np.mean(func_losses)
            avg_phys = np.mean(phys_losses)
            
            if avg_phys > 0:
                ratio = avg_func / avg_phys
                
                # Adjust lambda to balance the losses
                target_ratio = 1.0  # Target equal contribution
                if ratio > target_ratio * 2:  # Function loss too high
                    self.lambda_physics *= (1 + self.adaptation_rate)
                elif ratio < target_ratio / 2:  # Physics loss too high
                    self.lambda_physics *= (1 - self.adaptation_rate)
                
                # Clamp lambda to reasonable bounds
                self.lambda_physics = np.clip(
                    self.lambda_physics, self.min_lambda, self.max_lambda
                )


def create_loss_function(
    loss_type: str = 'standard',
    lambda_physics: float = 1.0,
    **kwargs
) -> PINNILoss:
    """Factory function to create loss functions
    
    Args:
        loss_type (str): Type of loss function ('standard' or 'adaptive')
        lambda_physics (float): Physics constraint weight
        **kwargs: Additional arguments for loss function
    
    Returns:
        PINNILoss: Configured loss function
    """
    if loss_type == 'standard':
        return PINNILoss(lambda_physics=lambda_physics, **kwargs)
    elif loss_type == 'adaptive':
        return AdaptivePINNILoss(lambda_physics=lambda_physics, **kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
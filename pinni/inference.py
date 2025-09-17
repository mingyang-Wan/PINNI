"""Inference and Integration Module for PINNI

Implements inference functionality for trained PINNI models,
including numerical integration using trapezoidal rule and stress tensor computation.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union, Dict, List
import time
from scipy import integrate

from .model import PINNIModel, StrainFeatureExtractor


class PINNIInference:
    """Inference engine for trained PINNI models
    
    Provides efficient computation of stress tensors through neural network
    integration, replacing traditional numerical constitutive integration.
    
    Args:
        model (PINNIModel): Trained PINNI model
        device (str): Device for inference ('cpu' or 'cuda')
        integration_method (str): Numerical integration method ('trapezoidal' or 'simpson')
    """
    
    def __init__(
        self,
        model: PINNIModel,
        device: Optional[str] = None,
        integration_method: str = 'trapezoidal'
    ):
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Move model to device and set to evaluation mode
        self.model = model.to(self.device)
        self.model.eval()
        
        self.integration_method = integration_method
        self.feature_extractor = StrainFeatureExtractor()
        
        # Performance tracking
        self.inference_times = []
        self.integration_times = []
        
        print(f"PINNI Inference engine initialized on {self.device}")
        print(f"Integration method: {integration_method}")
    
    def compute_stress(
        self,
        strain_tensor: Union[np.ndarray, torch.Tensor],
        integration_bounds: Tuple[float, float] = (0.0, 1.0),
        n_integration_points: int = 100,
        return_details: bool = False
    ) -> Union[float, Dict[str, Union[float, np.ndarray]]]:
        """Compute stress tensor from strain tensor using PINNI
        
        This is the main interface for stress computation during simulation.
        
        Args:
            strain_tensor (Union[np.ndarray, torch.Tensor]): Input strain tensor
            integration_bounds (Tuple[float, float]): Integration domain [a, b]
            n_integration_points (int): Number of integration points
            return_details (bool): Whether to return detailed computation info
        
        Returns:
            Union[float, Dict]: Computed stress value or detailed results
        """
        start_time = time.time()
        
        # Step 1: Extract features from strain tensor
        if isinstance(strain_tensor, torch.Tensor):
            strain_np = strain_tensor.cpu().numpy()
        else:
            strain_np = np.asarray(strain_tensor)
        
        strain_features = self.feature_extractor.extract_features(strain_np)
        strain_features_tensor = torch.tensor(strain_features, dtype=torch.float32, device=self.device)
        
        # Step 2: Generate integration points
        x_points = torch.linspace(
            integration_bounds[0],
            integration_bounds[1],
            n_integration_points,
            device=self.device
        )
        
        # Step 3: Batch neural network inference
        integrand_values = self._batch_inference(x_points, strain_features_tensor)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Step 4: Numerical integration
        integration_start = time.time()
        integral_value = self._numerical_integration(integrand_values, x_points)
        integration_time = time.time() - integration_start
        self.integration_times.append(integration_time)
        
        # Step 5: Compute final stress tensor
        stress_tensor = self._compute_final_stress(integral_value, strain_np)
        
        if return_details:
            return {
                'stress_tensor': stress_tensor,
                'integral_value': integral_value.item(),
                'integrand_values': integrand_values.cpu().numpy(),
                'integration_points': x_points.cpu().numpy(),
                'strain_features': strain_features,
                'inference_time': inference_time,
                'integration_time': integration_time,
                'total_time': inference_time + integration_time
            }
        else:
            return stress_tensor
    
    def _batch_inference(
        self,
        x_points: torch.Tensor,
        strain_features: torch.Tensor
    ) -> torch.Tensor:
        """Perform batch neural network inference
        
        Args:
            x_points (torch.Tensor): Integration points (N,)
            strain_features (torch.Tensor): Strain features (feature_dim,)
        
        Returns:
            torch.Tensor: Predicted integrand values (N,)
        """
        batch_size = len(x_points)
        
        # Expand strain features to match batch size
        strain_expanded = strain_features.unsqueeze(0).expand(batch_size, -1)
        
        # Reshape x_points to column vector
        x_reshaped = x_points.unsqueeze(1)
        
        # Concatenate inputs: [x, strain_features]
        model_inputs = torch.cat([x_reshaped, strain_expanded], dim=1)
        
        # Batch inference
        with torch.no_grad():
            integrand_values = self.model(model_inputs)
            if integrand_values.dim() > 1:
                integrand_values = integrand_values.squeeze(-1)
        
        return integrand_values
    
    def _numerical_integration(
        self,
        y_values: torch.Tensor,
        x_points: torch.Tensor
    ) -> torch.Tensor:
        """Perform numerical integration using specified method
        
        Args:
            y_values (torch.Tensor): Function values at integration points
            x_points (torch.Tensor): Integration points
        
        Returns:
            torch.Tensor: Approximated integral value
        """
        if self.integration_method == 'trapezoidal':
            return self._trapezoidal_integration(y_values, x_points)
        elif self.integration_method == 'simpson':
            return self._simpson_integration(y_values, x_points)
        else:
            raise ValueError(f"Unknown integration method: {self.integration_method}")
    
    def _trapezoidal_integration(
        self,
        y_values: torch.Tensor,
        x_points: torch.Tensor
    ) -> torch.Tensor:
        """Trapezoidal rule integration
        
        Implements: ∫f(x)dx ≈ Σ[i=0 to N-2] (f(x[i+1]) + f(x[i])) * (x[i+1] - x[i]) / 2
        """
        # Compute step sizes
        dx = x_points[1:] - x_points[:-1]
        
        # Trapezoidal rule
        integral = torch.sum(0.5 * (y_values[1:] + y_values[:-1]) * dx)
        
        return integral
    
    def _simpson_integration(
        self,
        y_values: torch.Tensor,
        x_points: torch.Tensor
    ) -> torch.Tensor:
        """Simpson's rule integration"""
        n = len(x_points)
        
        if n < 3:
            # Fall back to trapezoidal for insufficient points
            return self._trapezoidal_integration(y_values, x_points)
        
        # Ensure odd number of points for Simpson's rule
        if n % 2 == 0:
            # Use composite Simpson's + trapezoidal for last interval
            simpson_integral = self._composite_simpson(y_values[:-1], x_points[:-1])
            trap_integral = 0.5 * (y_values[-1] + y_values[-2]) * (x_points[-1] - x_points[-2])
            return simpson_integral + trap_integral
        else:
            return self._composite_simpson(y_values, x_points)
    
    def _composite_simpson(
        self,
        y_values: torch.Tensor,
        x_points: torch.Tensor
    ) -> torch.Tensor:
        """Composite Simpson's rule"""
        n = len(x_points)
        h = (x_points[-1] - x_points[0]) / (n - 1)
        
        # Simpson's rule weights
        integral = y_values[0] + y_values[-1]
        
        # Odd indices (weight 4)
        for i in range(1, n-1, 2):
            integral += 4 * y_values[i]
        
        # Even indices (weight 2)
        for i in range(2, n-1, 2):
            integral += 2 * y_values[i]
        
        integral *= h / 3
        return integral
    
    def _compute_final_stress(
        self,
        integral_value: torch.Tensor,
        strain_tensor: np.ndarray
    ) -> float:
        """Compute final stress tensor from integral value
        
        In this simplified implementation, we return the integral value directly.
        In practice, you might need additional transformations based on your
        specific constitutive model.
        
        Args:
            integral_value (torch.Tensor): Computed integral
            strain_tensor (np.ndarray): Original strain tensor
        
        Returns:
            float: Stress tensor component
        """
        # For demonstration, return the integral value as stress
        # In practice, you might apply additional transformations
        return integral_value.item()
    
    def batch_compute_stress(
        self,
        strain_tensors: List[Union[np.ndarray, torch.Tensor]],
        integration_bounds: Tuple[float, float] = (0.0, 1.0),
        n_integration_points: int = 100
    ) -> List[float]:
        """Compute stress tensors for multiple strain states
        
        Args:
            strain_tensors (List): List of strain tensors
            integration_bounds (Tuple[float, float]): Integration domain
            n_integration_points (int): Number of integration points
        
        Returns:
            List[float]: Computed stress values
        """
        stress_values = []
        
        for strain in strain_tensors:
            stress = self.compute_stress(
                strain, integration_bounds, n_integration_points
            )
            stress_values.append(stress)
        
        return stress_values
    
    def benchmark_performance(
        self,
        strain_tensor: Union[np.ndarray, torch.Tensor],
        n_runs: int = 100,
        n_integration_points: int = 100
    ) -> Dict[str, float]:
        """Benchmark inference performance
        
        Args:
            strain_tensor: Test strain tensor
            n_runs (int): Number of benchmark runs
            n_integration_points (int): Number of integration points
        
        Returns:
            Dict[str, float]: Performance statistics
        """
        print(f"Benchmarking performance over {n_runs} runs...")
        
        times = []
        
        for i in range(n_runs):
            start_time = time.time()
            _ = self.compute_stress(strain_tensor, n_integration_points=n_integration_points)
            end_time = time.time()
            times.append(end_time - start_time)
        
        times = np.array(times)
        
        stats = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'throughput_hz': 1.0 / np.mean(times)
        }
        
        print("Performance Statistics:")
        print(f"  Mean time: {stats['mean_time']*1000:.3f} ms")
        print(f"  Std time: {stats['std_time']*1000:.3f} ms")
        print(f"  Min time: {stats['min_time']*1000:.3f} ms")
        print(f"  Max time: {stats['max_time']*1000:.3f} ms")
        print(f"  Throughput: {stats['throughput_hz']:.1f} Hz")
        
        return stats
    
    def compare_with_traditional(
        self,
        strain_tensor: Union[np.ndarray, torch.Tensor],
        traditional_integrand: callable,
        integration_bounds: Tuple[float, float] = (0.0, 1.0),
        n_integration_points: int = 100
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Compare PINNI results with traditional numerical integration
        
        Args:
            strain_tensor: Input strain tensor
            traditional_integrand: Traditional integrand function
            integration_bounds: Integration domain
            n_integration_points: Number of integration points
        
        Returns:
            Dict: Comparison results
        """
        # PINNI computation
        pinni_start = time.time()
        pinni_result = self.compute_stress(
            strain_tensor, integration_bounds, n_integration_points, return_details=True
        )
        pinni_time = time.time() - pinni_start
        
        # Traditional computation
        traditional_start = time.time()
        traditional_integral, _ = integrate.quad(
            traditional_integrand,
            integration_bounds[0],
            integration_bounds[1]
        )
        traditional_time = time.time() - traditional_start
        
        # Compute errors
        absolute_error = abs(pinni_result['integral_value'] - traditional_integral)
        relative_error = absolute_error / abs(traditional_integral) if traditional_integral != 0 else float('inf')
        
        comparison = {
            'pinni_integral': pinni_result['integral_value'],
            'traditional_integral': traditional_integral,
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'pinni_time': pinni_time,
            'traditional_time': traditional_time,
            'speedup': traditional_time / pinni_time,
            'pinni_details': pinni_result
        }
        
        print("Comparison Results:")
        print(f"  PINNI integral: {comparison['pinni_integral']:.6f}")
        print(f"  Traditional integral: {comparison['traditional_integral']:.6f}")
        print(f"  Absolute error: {comparison['absolute_error']:.6e}")
        print(f"  Relative error: {comparison['relative_error']:.6e}")
        print(f"  PINNI time: {comparison['pinni_time']*1000:.3f} ms")
        print(f"  Traditional time: {comparison['traditional_time']*1000:.3f} ms")
        print(f"  Speedup: {comparison['speedup']:.1f}x")
        
        return comparison
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get accumulated performance statistics
        
        Returns:
            Dict[str, float]: Performance statistics
        """
        if not self.inference_times:
            return {'message': 'No inference data available'}
        
        inference_times = np.array(self.inference_times)
        integration_times = np.array(self.integration_times)
        total_times = inference_times + integration_times
        
        return {
            'total_inferences': len(self.inference_times),
            'mean_inference_time': np.mean(inference_times),
            'mean_integration_time': np.mean(integration_times),
            'mean_total_time': np.mean(total_times),
            'inference_throughput': 1.0 / np.mean(inference_times),
            'total_throughput': 1.0 / np.mean(total_times)
        }
    
    def reset_performance_stats(self):
        """Reset performance tracking statistics"""
        self.inference_times.clear()
        self.integration_times.clear()
        print("Performance statistics reset.")


class AdaptiveIntegration:
    """Adaptive integration with error control
    
    Provides adaptive integration that adjusts the number of integration
    points based on the desired accuracy.
    """
    
    def __init__(
        self,
        inference_engine: PINNIInference,
        tolerance: float = 1e-6,
        max_points: int = 1000,
        min_points: int = 10
    ):
        self.inference_engine = inference_engine
        self.tolerance = tolerance
        self.max_points = max_points
        self.min_points = min_points
    
    def adaptive_compute_stress(
        self,
        strain_tensor: Union[np.ndarray, torch.Tensor],
        integration_bounds: Tuple[float, float] = (0.0, 1.0)
    ) -> Dict[str, Union[float, int]]:
        """Compute stress with adaptive integration
        
        Args:
            strain_tensor: Input strain tensor
            integration_bounds: Integration domain
        
        Returns:
            Dict: Results including stress value and number of points used
        """
        n_points = self.min_points
        prev_result = None
        
        while n_points <= self.max_points:
            current_result = self.inference_engine.compute_stress(
                strain_tensor, integration_bounds, n_points
            )
            
            if prev_result is not None:
                error = abs(current_result - prev_result)
                if error < self.tolerance:
                    return {
                        'stress_value': current_result,
                        'n_points_used': n_points,
                        'estimated_error': error,
                        'converged': True
                    }
            
            prev_result = current_result
            n_points = min(n_points * 2, self.max_points)
        
        return {
            'stress_value': current_result,
            'n_points_used': n_points,
            'estimated_error': float('inf'),
            'converged': False
        }


def create_inference_engine(
    model_path: str,
    device: Optional[str] = None,
    **kwargs
) -> PINNIInference:
    """Factory function to create inference engine from saved model
    
    Args:
        model_path (str): Path to saved model
        device (str, optional): Device for inference
        **kwargs: Additional arguments for PINNIInference
    
    Returns:
        PINNIInference: Configured inference engine
    """
    # Load model
    model = PINNIModel.load_model(model_path, device)
    
    # Create inference engine
    return PINNIInference(model, device, **kwargs)
"""Utility Functions for PINNI

Provides visualization, model management, configuration handling,
and other utility functions for the PINNI framework.
"""

import os
import json
import pickle
import yaml
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
from pathlib import Path

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ConfigManager:
    """Configuration management for PINNI experiments
    
    Handles loading, saving, and validation of configuration files
    for reproducible experiments.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_default_config()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'model': {
                'input_dim': 7,  # 6 strain components + 1 integration variable
                'hidden_layers': [256, 256, 256, 256, 256],
                'activation': 'tanh',
                'output_dim': 1,
                'dropout_rate': 0.0
            },
            'training': {
                'batch_size': 1024,
                'learning_rate': 1e-3,
                'num_epochs': 1000,
                'optimizer': 'adam',
                'scheduler': 'cosine',
                'early_stopping_patience': 50,
                'validation_split': 0.2
            },
            'loss': {
                'function_weight': 1.0,
                'physics_weight': 1.0,
                'integration_method': 'trapezoidal',
                'adaptive_weights': False
            },
            'data': {
                'num_samples': 10000,
                'strain_range': [-0.1, 0.1],
                'integration_bounds': [0.0, 1.0],
                'noise_level': 0.01,
                'constitutive_model': 'linear_elastic'
            },
            'inference': {
                'integration_method': 'trapezoidal',
                'n_integration_points': 100,
                'device': 'auto'
            },
            'logging': {
                'log_level': 'INFO',
                'save_frequency': 10,
                'plot_frequency': 50
            }
        }
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from file
        
        Args:
            config_path (str): Path to configuration file (.json or .yaml)
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        file_ext = Path(config_path).suffix.lower()
        
        if file_ext == '.json':
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
        elif file_ext in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")
        
        # Merge with default config
        self._deep_update(self.config, loaded_config)
        self.config_path = config_path
        print(f"Configuration loaded from {config_path}")
    
    def save_config(self, save_path: str) -> None:
        """Save current configuration to file
        
        Args:
            save_path (str): Path to save configuration
        """
        file_ext = Path(save_path).suffix.lower()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if file_ext == '.json':
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        elif file_ext in ['.yaml', '.yml']:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")
        
        print(f"Configuration saved to {save_path}")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation
        
        Args:
            key_path (str): Dot-separated key path (e.g., 'model.hidden_layers')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation
        
        Args:
            key_path (str): Dot-separated key path
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value


class Visualizer:
    """Visualization utilities for PINNI
    
    Provides comprehensive plotting and visualization capabilities
    for training progress, model predictions, and analysis.
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Set up plotting parameters
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_name: Optional[str] = None
    ) -> None:
        """Plot training history
        
        Args:
            history (Dict): Training history with loss values
            save_name (str, optional): Name to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Total loss
        if 'train_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        
        # Function fitting loss
        if 'train_function_loss' in history:
            axes[0, 1].plot(history['train_function_loss'], label='Train Function Loss', linewidth=2)
        if 'val_function_loss' in history:
            axes[0, 1].plot(history['val_function_loss'], label='Val Function Loss', linewidth=2)
        axes[0, 1].set_title('Function Fitting Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        
        # Physics loss
        if 'train_physics_loss' in history:
            axes[1, 0].plot(history['train_physics_loss'], label='Train Physics Loss', linewidth=2)
        if 'val_physics_loss' in history:
            axes[1, 0].plot(history['val_physics_loss'], label='Val Physics Loss', linewidth=2)
        axes[1, 0].set_title('Physics Constraint Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].set_yscale('log')
        
        # Learning rate
        if 'learning_rate' in history:
            axes[1, 1].plot(history['learning_rate'], linewidth=2, color='orange')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = os.path.join(self.save_dir, f"{save_name}_training_history.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions(
        self,
        x_true: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Model Predictions",
        save_name: Optional[str] = None
    ) -> None:
        """Plot model predictions vs true values
        
        Args:
            x_true (np.ndarray): Input values
            y_true (np.ndarray): True output values
            y_pred (np.ndarray): Predicted output values
            title (str): Plot title
            save_name (str, optional): Name to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Predictions vs True values
        axes[0].scatter(y_true, y_pred, alpha=0.6, s=20)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        axes[0].set_xlabel('True Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Predictions vs True Values')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Error distribution
        errors = y_pred - y_true
        axes[1].hist(errors, bins=50, alpha=0.7, density=True, edgecolor='black')
        axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[1].set_xlabel('Prediction Error')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Error Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        r2 = 1 - np.sum(errors**2) / np.sum((y_true - np.mean(y_true))**2)
        
        stats_text = f'MAE: {mae:.6f}\nRMSE: {rmse:.6f}\nR²: {r2:.6f}'
        axes[1].text(0.05, 0.95, stats_text, transform=axes[1].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = os.path.join(self.save_dir, f"{save_name}_predictions.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Predictions plot saved to {save_path}")
        
        plt.show()
    
    def plot_integration_comparison(
        self,
        x_points: np.ndarray,
        integrand_values: np.ndarray,
        pinni_integral: float,
        true_integral: float,
        title: str = "Integration Comparison",
        save_name: Optional[str] = None
    ) -> None:
        """Plot integration comparison
        
        Args:
            x_points (np.ndarray): Integration points
            integrand_values (np.ndarray): Integrand values
            pinni_integral (float): PINNI computed integral
            true_integral (float): True integral value
            title (str): Plot title
            save_name (str, optional): Name to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Integrand function
        ax1.plot(x_points, integrand_values, 'b-', linewidth=2, label='PINNI Integrand')
        ax1.fill_between(x_points, 0, integrand_values, alpha=0.3, color='blue')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title('Integrand Function')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Integration comparison
        methods = ['PINNI', 'True']
        values = [pinni_integral, true_integral]
        colors = ['blue', 'red']
        
        bars = ax2.bar(methods, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Integral Value')
        ax2.set_title('Integration Results Comparison')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.6f}', ha='center', va='bottom', fontweight='bold')
        
        # Add error information
        error = abs(pinni_integral - true_integral)
        relative_error = error / abs(true_integral) if true_integral != 0 else float('inf')
        
        error_text = f'Absolute Error: {error:.6e}\nRelative Error: {relative_error:.6e}'
        ax2.text(0.02, 0.98, error_text, transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = os.path.join(self.save_dir, f"{save_name}_integration.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Integration comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_performance_benchmark(
        self,
        performance_data: Dict[str, List[float]],
        save_name: Optional[str] = None
    ) -> None:
        """Plot performance benchmark results
        
        Args:
            performance_data (Dict): Performance timing data
            save_name (str, optional): Name to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Performance Benchmark', fontsize=16, fontweight='bold')
        
        # Timing comparison
        methods = list(performance_data.keys())
        times = [np.mean(performance_data[method]) for method in methods]
        stds = [np.std(performance_data[method]) for method in methods]
        
        bars = axes[0].bar(methods, times, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('Time (seconds)')
        axes[0].set_title('Average Computation Time')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, time, std in zip(bars, times, stds):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + std,
                        f'{time:.4f}±{std:.4f}s', ha='center', va='bottom')
        
        # Throughput comparison
        throughputs = [1.0 / time for time in times]
        bars2 = axes[1].bar(methods, throughputs, alpha=0.7, edgecolor='black', color='orange')
        axes[1].set_ylabel('Throughput (Hz)')
        axes[1].set_title('Computation Throughput')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, throughput in zip(bars2, throughputs):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{throughput:.1f} Hz', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            save_path = os.path.join(self.save_dir, f"{save_name}_performance.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance benchmark plot saved to {save_path}")
        
        plt.show()


class ModelManager:
    """Model management utilities
    
    Handles saving, loading, and versioning of PINNI models
    with metadata and experiment tracking.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def save_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> str:
        """Save model with metadata
        
        Args:
            model (torch.nn.Module): Model to save
            model_name (str): Name for the model
            metadata (Dict, optional): Additional metadata
            optimizer (torch.optim.Optimizer, optional): Optimizer state
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler state
        
        Returns:
            str: Path to saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.models_dir, f"{model_name}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model state
        model_path = os.path.join(model_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
        
        # Save complete model (for inference)
        full_model_path = os.path.join(model_dir, "full_model.pth")
        torch.save(model, full_model_path)
        
        # Save optimizer and scheduler if provided
        if optimizer:
            optimizer_path = os.path.join(model_dir, "optimizer.pth")
            torch.save(optimizer.state_dict(), optimizer_path)
        
        if scheduler:
            scheduler_path = os.path.join(model_dir, "scheduler.pth")
            torch.save(scheduler.state_dict(), scheduler_path)
        
        # Save metadata
        model_metadata = {
            'model_name': model_name,
            'timestamp': timestamp,
            'model_architecture': str(model),
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'save_path': model_dir
        }
        
        if metadata:
            model_metadata.update(metadata)
        
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2, default=str)
        
        print(f"Model saved to {model_dir}")
        return model_dir
    
    def load_model(
        self,
        model_path: str,
        device: Optional[str] = None
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Load model with metadata
        
        Args:
            model_path (str): Path to model directory or .pth file
            device (str, optional): Device to load model on
        
        Returns:
            Tuple[torch.nn.Module, Dict]: Loaded model and metadata
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if os.path.isdir(model_path):
            # Load from model directory
            full_model_path = os.path.join(model_path, "full_model.pth")
            metadata_path = os.path.join(model_path, "metadata.json")
            
            if os.path.exists(full_model_path):
                model = torch.load(full_model_path, map_location=device)
            else:
                raise FileNotFoundError(f"Model file not found in {model_path}")
            
            # Load metadata
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
        else:
            # Load from .pth file
            model = torch.load(model_path, map_location=device)
            metadata = {}
        
        model.eval()
        print(f"Model loaded from {model_path}")
        return model, metadata
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all saved models
        
        Returns:
            List[Dict]: List of model information
        """
        models = []
        
        for item in os.listdir(self.models_dir):
            item_path = os.path.join(self.models_dir, item)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    models.append(metadata)
        
        # Sort by timestamp
        models.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return models
    
    def delete_model(self, model_path: str) -> None:
        """Delete a saved model
        
        Args:
            model_path (str): Path to model directory
        """
        if os.path.exists(model_path) and os.path.isdir(model_path):
            import shutil
            shutil.rmtree(model_path)
            print(f"Model deleted: {model_path}")
        else:
            print(f"Model not found: {model_path}")


class ExperimentLogger:
    """Experiment logging and tracking
    
    Provides comprehensive logging of experiments including
    hyperparameters, metrics, and results.
    """
    
    def __init__(self, log_dir: str = "experiments"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.current_experiment = None
    
    def start_experiment(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        description: str = ""
    ) -> str:
        """Start a new experiment
        
        Args:
            experiment_name (str): Name of the experiment
            config (Dict): Experiment configuration
            description (str): Experiment description
        
        Returns:
            str: Experiment ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{experiment_name}_{timestamp}"
        
        experiment_dir = os.path.join(self.log_dir, experiment_id)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save experiment metadata
        experiment_info = {
            'experiment_id': experiment_id,
            'experiment_name': experiment_name,
            'description': description,
            'start_time': timestamp,
            'config': config,
            'status': 'running'
        }
        
        info_path = os.path.join(experiment_dir, "experiment_info.json")
        with open(info_path, 'w') as f:
            json.dump(experiment_info, f, indent=2, default=str)
        
        self.current_experiment = {
            'id': experiment_id,
            'dir': experiment_dir,
            'info': experiment_info,
            'metrics': []
        }
        
        print(f"Started experiment: {experiment_id}")
        return experiment_id
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """Log metrics for current experiment
        
        Args:
            metrics (Dict): Metrics to log
            step (int, optional): Step number
        """
        if not self.current_experiment:
            print("No active experiment. Start an experiment first.")
            return
        
        timestamp = datetime.now().isoformat()
        metric_entry = {
            'timestamp': timestamp,
            'step': step,
            'metrics': metrics
        }
        
        self.current_experiment['metrics'].append(metric_entry)
        
        # Save metrics to file
        metrics_path = os.path.join(self.current_experiment['dir'], "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.current_experiment['metrics'], f, indent=2)
    
    def end_experiment(
        self,
        final_metrics: Optional[Dict[str, float]] = None,
        notes: str = ""
    ) -> None:
        """End current experiment
        
        Args:
            final_metrics (Dict, optional): Final experiment metrics
            notes (str): Additional notes
        """
        if not self.current_experiment:
            print("No active experiment to end.")
            return
        
        # Update experiment info
        self.current_experiment['info']['status'] = 'completed'
        self.current_experiment['info']['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_experiment['info']['notes'] = notes
        
        if final_metrics:
            self.current_experiment['info']['final_metrics'] = final_metrics
        
        # Save updated experiment info
        info_path = os.path.join(self.current_experiment['dir'], "experiment_info.json")
        with open(info_path, 'w') as f:
            json.dump(self.current_experiment['info'], f, indent=2, default=str)
        
        print(f"Experiment completed: {self.current_experiment['id']}")
        self.current_experiment = None


def setup_reproducibility(seed: int = 42) -> None:
    """Setup reproducible random seeds
    
    Args:
        seed (int): Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Reproducibility setup with seed: {seed}")


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Get optimal device for computation
    
    Args:
        prefer_gpu (bool): Whether to prefer GPU if available
    
    Returns:
        torch.device: Selected device
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def print_model_summary(model: torch.nn.Module) -> None:
    """Print model architecture summary
    
    Args:
        model (torch.nn.Module): Model to summarize
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    print(f"Architecture: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*50)
    print(model)
    print("="*50 + "\n")
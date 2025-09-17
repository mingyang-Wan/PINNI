"""Training Module for PINNI

Implements comprehensive training functionality for Physics-Informed Neural Network Integration,
including training loops, validation, model checkpointing, and progress monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import os
import json
from tqdm import tqdm
from datetime import datetime

from .model import PINNIModel
from .loss import PINNILoss, create_loss_function



class PINNITrainer:
    """Comprehensive trainer for PINNI models
    
    Handles training, validation, checkpointing, and monitoring of PINNI models
    with physics-informed loss functions.
    
    Args:
        model (PINNIModel): PINNI model to train
        loss_function (PINNILoss): Physics-informed loss function
        optimizer (torch.optim.Optimizer): Optimizer for training
        device (str): Device for training ('cpu' or 'cuda')
        checkpoint_dir (str): Directory for saving checkpoints
    """
    
    def __init__(
        self,
        model: PINNIModel,
        loss_function: Optional[PINNILoss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[str] = None,
        checkpoint_dir: str = "./checkpoints"
    ):
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Set up loss function
        if loss_function is None:
            self.loss_function = create_loss_function('standard', lambda_physics=1.0)
        else:
            self.loss_function = loss_function
        
        # Set up optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        else:
            self.optimizer = optimizer
        
        # Set up scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training state
        self.checkpoint_dir = checkpoint_dir
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'function_loss': [],
            'physics_loss': [],
            'learning_rate': []
        }
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"Trainer initialized on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train(
        self,
        X_func: torch.Tensor,
        y_func: torch.Tensor,
        strain_features: torch.Tensor,
        I_true: torch.Tensor,
        epochs: int = 1000,
        batch_size: int = 256,
        validation_split: float = 0.2,
        n_physics_points: int = 50,
        early_stopping_patience: int = 50,
        save_best_model: bool = True,
        plot_training: bool = True
    ) -> Dict[str, List[float]]:
        """Train the PINNI model
        
        Args:
            X_func (torch.Tensor): Function fitting inputs
            y_func (torch.Tensor): Function fitting targets
            strain_features (torch.Tensor): Strain features for each sample
            I_true (torch.Tensor): True integral values
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data for validation
            n_physics_points (int): Number of points for physics constraint
            early_stopping_patience (int): Patience for early stopping
            save_best_model (bool): Whether to save the best model
            plot_training (bool): Whether to plot training curves
        
        Returns:
            Dict[str, List[float]]: Training history
        """
        print(f"Starting training for {epochs} epochs...")
        
        # Prepare data
        train_loader, val_loader, x_physics, integration_bounds = self._prepare_data(
            X_func, y_func, strain_features, I_true, batch_size, validation_split, n_physics_points
        )
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            train_metrics = self._train_epoch(train_loader, x_physics, integration_bounds)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader, x_physics, integration_bounds)
            
            # Update learning rate
            self.scheduler.step(val_metrics['total_loss'])
            
            # Record metrics
            self._record_metrics(train_metrics, val_metrics, epoch)
            
            # Early stopping and model saving
            if self._check_early_stopping(val_metrics['total_loss'], early_stopping_patience):
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if save_best_model and val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self._save_checkpoint(epoch, is_best=True)
            
            # Progress reporting
            if (epoch + 1) % 10 == 0 or epoch == 0:
                self._print_progress(epoch, train_metrics, val_metrics)
        
        print("Training completed!")
        
        # Plot training curves
        if plot_training:
            self._plot_training_curves()
        
        # Save final model
        if save_best_model:
            self._save_checkpoint(epochs-1, is_best=False, is_final=True)
        
        return self.training_history
    
    def _prepare_data(
        self,
        X_func: torch.Tensor,
        y_func: torch.Tensor,
        strain_features: torch.Tensor,
        I_true: torch.Tensor,
        batch_size: int,
        validation_split: float,
        n_physics_points: int
    ) -> Tuple[DataLoader, DataLoader, torch.Tensor, Tuple[float, float]]:
        """Prepare training and validation data loaders"""
        # Move data to device
        X_func = X_func.to(self.device)
        y_func = y_func.to(self.device)
        strain_features = strain_features.to(self.device)
        I_true = I_true.to(self.device)
        
        # Split data
        n_samples = len(strain_features)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        # Calculate points per sample
        points_per_sample = len(X_func) // n_samples
        
        # Split indices
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n_samples))
        
        # Create training data
        train_func_indices = []
        for i in train_indices:
            start_idx = i * points_per_sample
            end_idx = (i + 1) * points_per_sample
            train_func_indices.extend(range(start_idx, end_idx))
        
        val_func_indices = []
        for i in val_indices:
            start_idx = i * points_per_sample
            end_idx = (i + 1) * points_per_sample
            val_func_indices.extend(range(start_idx, end_idx))
        
        # Repeat strain features and I_true to match function data
        train_strain_repeated = strain_features[train_indices].repeat_interleave(points_per_sample, dim=0)
        train_I_repeated = I_true[train_indices].repeat_interleave(points_per_sample)
        
        val_strain_repeated = strain_features[val_indices].repeat_interleave(points_per_sample, dim=0)
        val_I_repeated = I_true[val_indices].repeat_interleave(points_per_sample)
        
        # Create datasets
        train_dataset = TensorDataset(
            X_func[train_func_indices],
            y_func[train_func_indices],
            train_strain_repeated,
            train_I_repeated
        )
        
        val_dataset = TensorDataset(
            X_func[val_func_indices],
            y_func[val_func_indices],
            val_strain_repeated,
            val_I_repeated
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Generate physics constraint points
        x_physics = torch.linspace(0.0, 1.0, n_physics_points, device=self.device)
        integration_bounds = (0.0, 1.0)
        
        return train_loader, val_loader, x_physics, integration_bounds
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        x_physics: torch.Tensor,
        integration_bounds: Tuple[float, float]
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_metrics = {'total_loss': 0, 'function_loss': 0, 'physics_loss': 0}
        n_batches = 0
        
        for batch_data in train_loader:
            X_batch, y_batch, strain_batch, I_batch = batch_data
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            loss_dict = self.loss_function(
                model=self.model,
                x_func=X_batch,
                y_func=y_batch,
                x_physics=x_physics,
                strain_features=strain_batch[0],  # Use first strain in batch
                I_true=I_batch[0],  # Use first integral in batch
                integration_bounds=integration_bounds
            )
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Accumulate metrics
            for key in total_metrics:
                total_metrics[key] += loss_dict[key].item()
            n_batches += 1
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= n_batches
        
        return total_metrics
    
    def _validate_epoch(
        self,
        val_loader: DataLoader,
        x_physics: torch.Tensor,
        integration_bounds: Tuple[float, float]
    ) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_metrics = {'total_loss': 0, 'function_loss': 0, 'physics_loss': 0}
        n_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                X_batch, y_batch, strain_batch, I_batch = batch_data
                
                # Compute loss
                loss_dict = self.loss_function(
                    model=self.model,
                    x_func=X_batch,
                    y_func=y_batch,
                    x_physics=x_physics,
                    strain_features=strain_batch[0],
                    I_true=I_batch[0],
                    integration_bounds=integration_bounds
                )
                
                # Accumulate metrics
                for key in total_metrics:
                    total_metrics[key] += loss_dict[key].item()
                n_batches += 1
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= n_batches
        
        return total_metrics
    
    def _record_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int
    ):
        """Record training metrics"""
        self.training_history['train_loss'].append(train_metrics['total_loss'])
        self.training_history['val_loss'].append(val_metrics['total_loss'])
        self.training_history['function_loss'].append(train_metrics['function_loss'])
        self.training_history['physics_loss'].append(train_metrics['physics_loss'])
        self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
    
    def _check_early_stopping(self, val_loss: float, patience: int) -> bool:
        """Check early stopping condition"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            return self.epochs_without_improvement >= patience
    
    def _print_progress(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Print training progress"""
        lr = self.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:4d} | "
              f"Train Loss: {train_metrics['total_loss']:.6f} | "
              f"Val Loss: {val_metrics['total_loss']:.6f} | "
              f"Func Loss: {train_metrics['function_loss']:.6f} | "
              f"Phys Loss: {train_metrics['physics_loss']:.6f} | "
              f"LR: {lr:.2e}")
    
    def _save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        is_final: bool = False
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'model_config': self.model.get_model_info()
        }
        
        if is_best:
            filepath = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, filepath)
            print(f"Best model saved at epoch {epoch+1}")
        
        if is_final:
            filepath = os.path.join(self.checkpoint_dir, 'final_model.pth')
            torch.save(checkpoint, filepath)
            print(f"Final model saved at epoch {epoch+1}")
    
    def _plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Total loss
        axes[0, 0].plot(self.training_history['train_loss'], label='Train')
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Function loss
        axes[0, 1].plot(self.training_history['function_loss'])
        axes[0, 1].set_title('Function Fitting Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Physics loss
        axes[1, 0].plot(self.training_history['physics_loss'])
        axes[1, 0].set_title('Physics Constraint Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(self.training_history['learning_rate'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.checkpoint_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training curves saved to {plot_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model from checkpoint
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        
        Returns:
            Dict[str, Any]: Checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        return checkpoint
    
    def evaluate_model(
        self,
        test_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate model on test data
        
        Args:
            test_data: Tuple of (X_func, y_func, strain_features, I_true)
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        X_func, y_func, strain_features, I_true = test_data
        
        # Move to device
        X_func = X_func.to(self.device)
        y_func = y_func.to(self.device)
        strain_features = strain_features.to(self.device)
        I_true = I_true.to(self.device)
        
        self.model.eval()
        
        with torch.no_grad():
            # Function prediction accuracy
            y_pred = self.model(X_func)
            if y_pred.dim() > 1:
                y_pred = y_pred.squeeze(-1)
            
            mse = torch.mean((y_pred - y_func) ** 2)
            mae = torch.mean(torch.abs(y_pred - y_func))
            
            # R² score
            ss_res = torch.sum((y_func - y_pred) ** 2)
            ss_tot = torch.sum((y_func - torch.mean(y_func)) ** 2)
            r2 = 1 - ss_res / ss_tot
        
        metrics = {
            'mse': mse.item(),
            'mae': mae.item(),
            'r2_score': r2.item(),
            'rmse': torch.sqrt(mse).item()
        }
        
        print("Evaluation Results:")
        for key, value in metrics.items():
            print(f"  {key.upper()}: {value:.6f}")
        
        return metrics


def create_trainer(
    model: PINNIModel,
    loss_type: str = 'standard',
    optimizer_type: str = 'adam',
    learning_rate: float = 1e-3,
    **kwargs
) -> PINNITrainer:
    """Factory function to create PINNI trainer
    
    Args:
        model (PINNIModel): PINNI model
        loss_type (str): Type of loss function
        optimizer_type (str): Type of optimizer
        learning_rate (float): Learning rate
        **kwargs: Additional arguments
    
    Returns:
        PINNITrainer: Configured trainer
    """
    # Create loss function
    loss_function = create_loss_function(loss_type, **kwargs)
    
    # Create optimizer
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return PINNITrainer(model, loss_function, optimizer, **kwargs)
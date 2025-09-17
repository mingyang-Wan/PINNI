#!/usr/bin/env python3
"""
Model Evaluation Script for PINNI

This script evaluates the performance of trained PINNI models.
Users should provide their own dataset following the format specified in README.md.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from pinni.model import PINNIModel
from pinni.loss import PINNILoss
from pinni.inference import PINNIInference

class ModelEvaluator:
    """PINNI Model Evaluator
    
    Provides comprehensive model performance evaluation functionality, including various error metric calculations and visualization.
    
    Args:
        model_path (str): Path to trained model
        device (str): Computing device ('cpu' or 'cuda')
    """
    
    def __init__(self, model_path=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model_path = model_path
        self.model = None
        self.inference_engine = None
        
        if model_path and os.path.exists(model_path):
            self.model = self._load_model(model_path)
        else:
            self.model = self._create_test_model()
    
    def _load_model(self, model_path):
        """Load trained model"""
        print(f"📦 Loading model: {model_path}")
        
        # Create model instance
        model = PINNIModel().to(self.device)
        
        # Load model weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return None
            
        print("✅ Model loaded successfully")
        return model
    
    def _create_test_model(self):
        """Create a test untrained model"""
        print("🔧 Creating test model (untrained)")
        model = PINNIModel().to(self.device)
        model.eval()
        print("✅ Test model created successfully")
        return model
    
    def load_test_data(self, data_path):
        """Load test data from file"""
        print(f"📊 Loading test data from: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Test data file not found: {data_path}")
        
        # Load data
        data = np.load(data_path)
        
        # Extract components
        X_test = torch.FloatTensor(data['x']).to(self.device)
        y_test = torch.FloatTensor(data['function_values']).to(self.device)
        I_test = torch.FloatTensor(data['integral_values']).to(self.device)
        strain_features = torch.FloatTensor(data['strain']).to(self.device)
        
        print("✅ Test data loading completed")
        return X_test, y_test, I_test, strain_features
    
    def evaluate_function_fitting(self, X_test, y_test):
        """Evaluate function fitting performance"""
        print("🎯 Evaluating function fitting performance...")
        
        self.model.eval()
        # Batch prediction
        with torch.no_grad():
            y_pred = self.model(X_test)
            
        # Convert to numpy
        y_true = y_test.cpu().numpy().flatten()
        y_pred = y_pred.cpu().numpy().flatten()
        
        # Calculate various error metrics
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        max_error = np.max(np.abs(y_true - y_pred))
        std_error = np.std(y_true - y_pred)
        
        return {
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'max_error': max_error,
            'std_error': std_error,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def evaluate_integration_accuracy(self, strain_features, I_test):
        """Evaluate integration calculation accuracy"""
        print("🧮 Evaluating integration calculation accuracy...")
        
        # Convert strain features to 3D strain tensor (reverse engineering)
        # Simplified processing here, actual applications need more precise inverse transformation
        strain_tensors = strain_features[:, :3].numpy()  # Take first 3 components as strain tensor
        
        integration_predictions = []
        integration_true = I_test.cpu().numpy()
        
        for i, strain_tensor in enumerate(strain_tensors):
            print(f"   Processing sample {i+1}/{len(strain_tensors)}")
            
            try:
                # Use inference engine to calculate integration
                if self.inference_engine is None:
                    self.inference_engine = PINNIInference(model=self.model, device=self.device)
                
                # Calculate stress (integration result)
                stress = self.inference_engine.compute_stress(strain_tensor)
                integration_predictions.append(stress[0] if isinstance(stress, np.ndarray) else stress)
                
            except Exception as e:
                print(f"   Warning: Sample {i} integration calculation failed: {e}")
                integration_predictions.append(0.0)  # Use default value
        
        integration_predictions = np.array(integration_predictions)
        
        # Calculate integration error metrics
        int_r2 = r2_score(integration_true, integration_predictions)
        int_mae = mean_absolute_error(integration_true, integration_predictions)
        int_rmse = np.sqrt(mean_squared_error(integration_true, integration_predictions))
        int_mape = np.mean(np.abs((integration_true - integration_predictions) / (integration_true + 1e-8))) * 100
        int_max_error = np.max(np.abs(integration_true - integration_predictions))
        
        return {
            'integration_r2': int_r2,
            'integration_mae': int_mae,
            'integration_rmse': int_rmse,
            'integration_mape': int_mape,
            'integration_max_error': int_max_error,
            'integration_true': integration_true,
            'integration_pred': integration_predictions
        }
    
    def plot_results(self, func_metrics, int_metrics, save_path):
        """Plot evaluation results"""
        print("📈 Generating evaluation result charts...")
        
        # Create save directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Set plot style
        plt.style.use('default')
        plt.rcParams['font.size'] = 10
        
        # 1. Function fitting results comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('PINNI Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # Function fitting scatter plot
        axes[0, 0].scatter(func_metrics['y_true'], func_metrics['y_pred'], alpha=0.6, s=20)
        axes[0, 0].plot([func_metrics['y_true'].min(), func_metrics['y_true'].max()], 
                       [func_metrics['y_true'].min(), func_metrics['y_true'].max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title(f'Function Fitting Comparison\nR² = {func_metrics["r2_score"]:.4f}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Function fitting error distribution
        func_errors = func_metrics['y_true'] - func_metrics['y_pred']
        axes[0, 1].hist(func_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Prediction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'Function Fitting Error Distribution\nMAE = {func_metrics["mae"]:.6f}')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Integration calculation scatter plot
        axes[1, 0].scatter(int_metrics['integration_true'], int_metrics['integration_pred'], alpha=0.6, s=20)
        axes[1, 0].plot([int_metrics['integration_true'].min(), int_metrics['integration_true'].max()], 
                       [int_metrics['integration_true'].min(), int_metrics['integration_true'].max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('True Integration Values')
        axes[1, 0].set_ylabel('Predicted Integration Values')
        axes[1, 0].set_title(f'Integration Calculation Comparison\nR² = {int_metrics["integration_r2"]:.4f}')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Integration calculation error distribution
        int_errors = int_metrics['integration_true'] - int_metrics['integration_pred']
        axes[1, 1].hist(int_errors, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Integration Prediction Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'Integration Error Distribution\nMAE = {int_metrics["integration_mae"]:.6f}')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Relative error analysis
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Function fitting relative error
        func_rel_errors = np.abs((func_metrics['y_true'] - func_metrics['y_pred']) / (func_metrics['y_true'] + 1e-8)) * 100
        axes[0].hist(func_rel_errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[0].set_xlabel('Relative Error (%)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Function Fitting Relative Error Distribution\nMean Relative Error = {func_metrics["mape"]:.2f}%')
        axes[0].grid(True, alpha=0.3)
        
        # Integration calculation relative error
        int_rel_errors = np.abs((int_metrics['integration_true'] - int_metrics['integration_pred']) / (int_metrics['integration_true'] + 1e-8)) * 100
        axes[1].hist(int_rel_errors, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1].set_xlabel('Relative Error (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Integration Calculation Relative Error Distribution\nMean Relative Error = {int_metrics["integration_mape"]:.2f}%')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        rel_error_path = save_path.replace('.png', '_relative_errors.png')
        plt.savefig(rel_error_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Charts saved to: {save_path}")
    
    def print_summary(self, func_metrics, int_metrics):
        """Print evaluation results summary"""
        print("\n" + "="*60)
        print("📊 PINNI Model Evaluation Results Summary")
        print("="*60)
        
        print("\n🎯 Function Fitting Performance:")
        print(f"   Coefficient of Determination (R²):        {func_metrics['r2_score']:.6f}")
        print(f"   Mean Absolute Error (MAE):   {func_metrics['mae']:.6f}")
        print(f"   Root Mean Square Error (RMSE):    {func_metrics['rmse']:.6f}")
        print(f"   Mean Absolute Percentage Error:   {func_metrics['mape']:.2f}%")
        print(f"   Maximum Error:            {func_metrics['max_error']:.6f}")
        print(f"   Error Standard Deviation:          {func_metrics['std_error']:.6f}")
        
        print("\n🧮 Integration Calculation Performance:")
        print(f"   Coefficient of Determination (R²):        {int_metrics['integration_r2']:.6f}")
        print(f"   Mean Absolute Error (MAE):   {int_metrics['integration_mae']:.6f}")
        print(f"   Root Mean Square Error (RMSE):    {int_metrics['integration_rmse']:.6f}")
        print(f"   Mean Absolute Percentage Error:   {int_metrics['integration_mape']:.2f}%")
        print(f"   Maximum Error:            {int_metrics['integration_max_error']:.6f}")
        
        print("\n📈 Performance Rating:")
        # Give performance rating based on R²
        func_grade = self._get_performance_grade(func_metrics['r2_score'])
        int_grade = self._get_performance_grade(int_metrics['integration_r2'])
        
        print(f"   Function Fitting Performance: {func_grade}")
        print(f"   Integration Calculation Performance: {int_grade}")
        
        print("="*60)
    
    def _get_performance_grade(self, r2_score):
        """Give performance rating based on R² score"""
        if r2_score >= 0.95:
            return "Excellent (R² ≥ 0.95)"
        elif r2_score >= 0.90:
            return "Good (0.90 ≤ R² < 0.95)"
        elif r2_score >= 0.80:
            return "Fair (0.80 ≤ R² < 0.90)"
        elif r2_score >= 0.60:
            return "Poor (0.60 ≤ R² < 0.80)"
        else:
            return "Very Poor (R² < 0.60)"
    
    def run_evaluation(self, n_test_samples=1000, n_integration_points=100, save_plots=True, save_dir="evaluation_results"):
        """Run complete model evaluation"""
        print("🚀 Starting PINNI model complete evaluation")
        print(f"   Test samples: {n_test_samples}")
        print(f"   Integration points: {n_integration_points}")
        
        if self.model is None:
            print("⚠️ No model loaded, will create test model")
            self.model = self._create_test_model()
        
        # 1. Generate test data
        X_test, y_test, I_test, strain_features = self.generate_test_data(
            n_test_samples, n_integration_points
        )
        
        # 2. Evaluate function fitting performance
        func_metrics = self.evaluate_function_fitting(X_test, y_test)
        
        # 3. Evaluate integration calculation accuracy
        int_metrics = self.evaluate_integration_accuracy(strain_features, I_test)
        
        # 4. Print results summary
        self.print_summary(func_metrics, int_metrics)
        
        # 5. Generate visualization results
        if save_plots:
            save_path = os.path.join(save_dir, "evaluation_results.png")
            self.plot_results(func_metrics, int_metrics, save_path)
            
            # Save numerical results
            results = {
                'function_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                   for k, v in func_metrics.items() if k not in ['y_true', 'y_pred']},
                'integration_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                      for k, v in int_metrics.items() if k not in ['integration_true', 'integration_pred']}
            }
            
            results_path = os.path.join(save_dir, "evaluation_metrics.json")
            os.makedirs(save_dir, exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"📁 Results saved to: {save_dir}")
        
        return func_metrics, int_metrics

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Evaluate PINNI model performance')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model')
    parser.add_argument('--test-samples', type=int, default=1000,
                       help='Number of test samples')
    parser.add_argument('--integration-points', type=int, default=100,
                       help='Number of integration points')
    parser.add_argument('--save-dir', type=str, default='evaluation_results',
                       help='Results save directory')
    parser.add_argument('--no-plots', action='store_true',
                       help='Do not generate charts')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(model_path=args.model_path)
    
    # Run evaluation
    func_metrics, int_metrics = evaluator.run_evaluation(
        n_test_samples=args.test_samples,
        n_integration_points=args.integration_points,
        save_plots=not args.no_plots,
        save_dir=args.save_dir
    )
    
    print("\n🎉 Model evaluation completed!")

if __name__ == "__main__":
    main()
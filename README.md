# PINNI

## Project Overview

Code base for paper: [TBD]

## Mathematical Principles

### Mechanical Constitutive Relations

Mechanical constitutive relations can be written in the following form:

```
σ = ∫ g(x; ε) dx
```

Where:
- `σ` is the stress tensor
- `x` is the integration variable
- `ε` is the strain tensor
- `g(x; ε)` is the integrand function

### PINNI Method

PINNI uses neural networks to approximate the integrand function:

```
g(x; ε) ≈ NN(x, ε; θ)
```

### Loss Function

The total loss function consists of two parts:

```
L_total = L_func + λ * L_physic
```

- `L_func`: Function fitting loss (MSE)
- `L_physic`: Physics constraint loss (integration error)
- `λ`: Balance hyperparameter

## Project Structure

```
pinni/
├── pinni/
│   ├── model.py          # PINNI neural network model
│   ├── loss.py           # Physics-informed loss functions
│   ├── trainer.py        # Training module
│   ├── inference.py      # Inference and integration calculation
│   └── utils.py          # Utility functions
├── configs/
│   └── default_config.yaml # Default configuration
├── evaluate_model.py     # Model evaluation
└── requirements.txt      # Dependencies

```

## Dataset Format Requirements

To use PINNI effectively, your dataset should follow these specifications:

### Training Data Format

The training dataset should contain pairs of integration variables and corresponding function values:

```python
# Input format for training
training_data = {
    'x': np.array([...]),           # Integration variable (shape: [N, 1])
    'strain': np.array([...]),      # Strain tensor components (shape: [N, 6])
    'function_values': np.array([...]), # Target function g(x; ε) values (shape: [N, 1])
    'integral_values': np.array([...])  # True integral values for physics constraint (shape: [M, 1])
}
```

### Data Structure Details

1. **Integration Variable (x)**:
   - Range: [0, 1] (normalized)
   - Type: Float32
   - Shape: [batch_size, 1]

2. **Strain Tensor Components**:
   - Components: [ε₁₁, ε₂₂, ε₃₃, ε₁₂, ε₁₃, ε₂₃]
   - Type: Float32
   - Shape: [batch_size, 6]
   - Normalization: Recommended to normalize to [-1, 1] range

3. **Function Values**:
   - Target values for g(x; ε)
   - Type: Float32
   - Shape: [batch_size, 1]

4. **Integral Values**:
   - Ground truth integral results for physics constraint
   - Type: Float32
   - Shape: [batch_size, 1]

### Example Data Loading

```python
import numpy as np
import torch

def load_dataset(data_path):
    """
    Load and preprocess dataset for PINNI training
    
    Args:
        data_path: Path to dataset file (.npz format recommended)
    
    Returns:
        Dictionary containing processed training data
    """
    data = np.load(data_path)
    
    return {
        'x': torch.FloatTensor(data['x']),
        'strain': torch.FloatTensor(data['strain']),
        'function_values': torch.FloatTensor(data['function_values']),
        'integral_values': torch.FloatTensor(data['integral_values'])
    }
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Dataset

Ensure your dataset follows the format specified in the [Dataset Format Requirements](#dataset-format-requirements) section.

### 2. Train PINNI Model

```python
from pinni.model import PINNIModel
from pinni.trainer import PINNITrainer
import torch

# Load your dataset
def load_dataset(data_path):
    data = np.load(data_path)
    return {
        'x': torch.FloatTensor(data['x']),
        'strain': torch.FloatTensor(data['strain']),
        'function_values': torch.FloatTensor(data['function_values']),
        'integral_values': torch.FloatTensor(data['integral_values'])
    }

# Create and train model
model = PINNIModel()
trainer = PINNITrainer(model)

# Load your training data
training_data = load_dataset('path/to/your/dataset.npz')
trainer.train(training_data)
```

### 3. Use Trained Model for Inference

```python
from pinni.inference import PINNIInference

# Create inference engine
inference = PINNIInference(model_path='path/to/model.pth')

# Calculate stress tensor
strain_tensor = np.array([0.1, 0.05, 0.02, 0.01, 0.005, 0.002])  # 6-component strain tensor
stress = inference.compute_stress(strain_tensor)
```

## Model Architecture

- **Input Layer**: Integration variable x + strain tensor features
- **Hidden Layers**: 5 layers, 256 neurons each
- **Activation Function**: Tanh (ensures integrability)
- **Output Layer**: Single output (integrand function value)

## Training Strategy

1. **Data Preparation**: Generate true function values and corresponding integral values for integration points
2. **Dual Constraints**: Simultaneously optimize point-wise fitting and integration constraints
3. **Physical Consistency**: Validate integration accuracy through trapezoidal rule

## License

This project is licensed under the MIT License.

## Citation

If you use this project in your research, please cite [TBD].

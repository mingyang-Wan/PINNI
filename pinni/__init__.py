"""
PINNI - Physics-Informed Neural-Network Integration

A physics-informed neural network framework for accelerating mechanical 
constitutive integration calculations in dynamic fracture simulations.
"""

from .model import PINNIModel
from .loss import PINNILoss
from .trainer import PINNITrainer
from .inference import PINNIInference
from .utils import *

__version__ = "1.0.0"

__all__ = [
    "PINNIModel",
    "PINNILoss", 
    "PINNITrainer",
    "PINNIInference",
]

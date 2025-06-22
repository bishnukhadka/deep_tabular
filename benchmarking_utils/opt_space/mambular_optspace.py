from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelConfig:
    n_layers: Tuple = ("int", 8, 32, 8)
    dropout: Tuple = ("uniform", 0.1, 0.5)

@dataclass
class TrainingConfig: 
    lr: Tuple = ("loguniform", 1e-5, 0.01)

@dataclass
class MambularOptSpace:
    model: ModelConfig
    training: TrainingConfig
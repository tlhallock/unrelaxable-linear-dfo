from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class Evaluation:
    x: np.ndarray
    objective: float
    constraints: np.ndarray
    failure: bool
    
    def is_feasible(self):
        return (
            np.all(np.isfinite(self.constraints)) and
            np.all(self.constraints <= 0)
        )

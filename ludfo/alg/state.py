from dataclasses import dataclass
from typing import Any
import numpy as np

from ludfo.alg.params import Params


@dataclass
class State:
    params: Params
    problem: Any
    
    iteration: int
    iterate: np.ndarray
    
    

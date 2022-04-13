from dataclasses import dataclass
import numpy as np


@dataclass
class LinearConstraints:
    '''
        Represents constraints of the form Ax <= b
    '''
    A: np.ndarray
    b: np.ndarray

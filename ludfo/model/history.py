from dataclasses import dataclass
import numpy as np

from model.bounds import Bounds
from model.evaluation import Evaluation


@dataclass
class IndexedEvaluation:
    idx: int
    evaluation: Evaluation


@dataclass
class History:
    bounds: Bounds
    evaluations: list[IndexedEvaluation]
    # sample regions
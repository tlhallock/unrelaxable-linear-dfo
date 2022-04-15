from dataclasses import dataclass
import numpy as np

from ludfo.model.bounds import Bounds
from ludfo.model.evaluation import Evaluation


@dataclass
class IndexedEvaluation:
    idx: int
    evaluation: Evaluation


@dataclass
class History:
    bounds: Bounds
    evaluations: list[IndexedEvaluation]
    # sample regions
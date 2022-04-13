
import itertools
from typing import Any, Iterator, Sequence, Tuple
import numpy as np
import math



def powers(dim: int, deg: int) -> Iterator[Sequence[int]]:
    for prod in itertools.combinations_with_replacement(range(dim), deg):
        yield prod


def quadratic_basis(n: int) -> Iterator[Tuple[float, Sequence[int]]]:
    for p in powers(n, 0):
        yield 1.0, p
    for p in powers(n, 1):
        yield 1.0, p
    for p in powers(n, 2):
        yield 0.5, p


def construct_vandermonde(sample: np.ndarray) -> np.ndarray:
    return np.array([
        [c * math.prod(x[i] for i in p) for c, p in quadratic_basis(sample.shape[1])]
        for x in sample
    ])


def get_quadratic_basis_dim(dim: int) -> int:
    return 1 + dim + dim * (dim + 1) // 2


import itertools
from typing import Any, Iterator, Sequence, Tuple
import numpy as np
import math

__all__ = ("construct_vandermonde", "get_quadratic_basis_dim")


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


def _construct_index_cache(dim: int) -> np.ndarray:
    ret = np.empty((dim, dim), dtype=np.int64)
    for idx, p in enumerate(powers(dim, 2)):
        ret[p[0], p[1]] = ret[p[1], p[0]] = 1 + dim + idx
    return ret


_idx_cache = {}


def _get_idx_cache(dim: int) -> np.ndarray:
    if dim not in _idx_cache:
        _idx_cache[dim] = _construct_index_cache(dim)
    return _idx_cache[dim]
    


def coef_to_matrix(dim, coef):
    c = ceof[0]
    g = ceof[1:(1+dim)]
    q = np.take(coef, _get_idx_cache) * np.diag([2.0] * dim)




def construct_vandermonde(sample: np.ndarray) -> np.ndarray:
    return np.array([
        [c * math.prod(x[i] for i in p) for c, p in quadratic_basis(sample.shape[1])]
        for x in sample
    ])


def get_quadratic_basis_dim(dim: int) -> int:
    return 1 + dim + dim * (dim + 1) // 2


if __name__ == '__main__':
    _construct_index_cache(3)

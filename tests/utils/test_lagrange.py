import numpy as np
from ludfo.model.ellipsoid import Ellipsoid

from ludfo.model.evaluation import Evaluation
from ludfo.model.history import IndexedEvaluation

from ludfo.utils.lagrange import (
    LagrangeParams,
    perform_lu_factorization,
)

def test_lagrange():
    dim = 2
    params = LagrangeParams(
        xsi_replace=1e-4,
        sample_region=Ellipsoid.create(
            q=np.eye(dim),
            center=np.zeros(dim),
            r=1,
        ),
        idx_evaluations=[
            IndexedEvaluation(
                idx=0,
                evaluation=Evaluation(
                    x=np.zeros(dim),
                    objective=3,
                    constraints=np.array([4, 3]),
                    failure=False,
                )
            )
        ]
    )
    perform_lu_factorization(params)
    
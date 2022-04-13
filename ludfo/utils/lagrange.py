
from typing import Any
import numpy as np

import logging

from dataclasses import dataclass
from model.history import IndexedEvaluation

from utils.trust_region import solve_tr_subproblem

from utils.basis import get_quadratic_basis_dim
from utils.basis import construct_vandermonde


@dataclass
class LagrangeParams:
    xsi_replace: float
    sample_region: Any
    idx_evaluations: list[IndexedEvaluation]
    
    def construct_cert(self):
        original_sample = np.array([
            evaluation.evaluation.x for evaluation in self.idx_evaluations])
        sample_indices = [evaluation.idx for evaluation in self.idx_evaluations]
        dim = original_sample.shape[1]
        basis_dim = get_quadratic_basis_dim(dim)
        orig_num_points = original_sample.shape[0]
        num_missing_vals = basis_dim - orig_num_points
        if num_missing_vals > 0:
            original_sample = np.vstack([
                original_sample,
                np.repeat(original_sample[0][np.newaxis, :], axis=0, repeats=num_missing_vals)
            ])
            sample_indices += [sample_indices[0]] * num_missing_vals
        
        num_points = original_sample.shape[0]
        current_sample = self.sample_region.shift_sample(original_sample)
        
        return Certification(
            params=self,
            success=True,
            num_points=num_points,
            basis_dimension=basis_dim,
            values=construct_vandermonde(current_sample),
            polys=np.eye(basis_dim),
            current_sample=current_sample,
            sample_indices=sample_indices,
            
            Lambda=None,
            quadratics=None,
            original_sample=original_sample,
        )


@dataclass
class Certification:
    params: LagrangeParams
    success: bool
    
    num_points: int
    basis_dimension: int
    values: np.ndarray
    polys: np.ndarray
    
    current_sample: np.ndarray
    sample_indices: list[int]
    
    Lambda: Any
    quadratics: Any
    
    original_sample: np.ndarray
    

    def get_shifted_points(self):
        return self.current_sample

    def get_lambdas(self):
        return self.polys.copy()

    def shifted_info(self):
        for idx in range(self.basis_dimension):
            yield (
                self.sample_indices[idx],
                self.current_sample[idx],
                self.quadratics[idx]
            )

    def add_to_plot(self, p):
        # self.params.feasibility_enforcer.add_to_plot(p)
        p.add_points(self.original_sample.sample_points, color='r', marker='x', label='original points')
        p.add_points(self.params.trust_region.unshift_points(self.current_sample.sample_points[:self.num_points]), color='b', marker='o', s=50, label='current points')



def _test_v(c):
    pass
    # V = MultiIndex.construct_vandermonde(c.params.basis, c.get_shifted_points())
    # lambdas = c.polys
    # reduced = c.values
    # ratio = np.linalg.norm(V@lambdas.T - reduced) / np.linalg.norm(reduced)
    # if ratio > 1e-10:
    #     print("TODO: The error here should not be so big...")
    # assert ratio < 1e-2, 'Failure to maintain structure of V: ' + str(ratio)


def _find_replacement(c, coef, params, name):
    pass
    # poly = Polynomial.construct_polynomial(params.basis, coef)
    # quad = poly.to_matrix_form()
    # success, ppoint, _ = solve_tr_subproblem(quad.g, 2 * quad.Q)
    # assert success, 'Unable to optimize lagrange polynomial'

    # success, npoint, _ = solve_tr_subproblem(-quad.g, -2 * quad.Q)
    # assert success, 'Unable to optimize lagrange negative polynomial'

    # pval = quad.evaluate(ppoint)
    # nval = quad.evaluate(npoint)

    # if params.plot_maximizations:
    #     p = params.plotter.create_plot(
    #         'lagrange_maximization_' + name,
    #         Bounds.create([-1.2, -1.2], [1.2, 1.2]),
    #         str(poly),
    #         subfolder='lagrange')
    #     p.add_point(npoint, label='negative minimizer', s=50, color='red')
    #     p.add_point(ppoint, label='positive minimizer', s=50, color='green')
    #     p.add_contour(quad.evaluate, label='polynomial', color='blue')
    #     p.add_circle(center=np.zeros_like(ppoint), radius=1, label='trust region')
    #     p.save()

    # if abs(nval) > abs(pval):
    #     return npoint, nval
    # else:
    #     return ppoint, pval

    # params.logger.log_message("Maximum absolute value of " + str(poly) + " over tr is " + str(mval))
    # return mpoint, mval


def _swap_rows(c, i1, i2):
    if i1 == i2:
        return

    t = c.sample_indices[i1]
    c.sample_indices[i1] = c.sample_indices[i2]
    c.sample_indices[i2] = t

    t = c.values[i1].copy()
    c.values[i1] = c.values[i2]
    c.values[i2] = t

    t = c.current_sample[i1].copy()
    c.current_sample[i1] = c.current_sample[i2]
    c.current_sample[i2] = t


def _replace_point(c, idx, shifted_point):
    c.values[idx, :] = (
        np.array([b.as_exponent(shifted_point) for b in c.params.basis]) @ c.polys.T
    )
    c.current_sample[idx] = shifted_point
    c.sample_indices[idx] = -1


def _reduce(c, i):
    pivot = c.values[i, i]

    c.values[:, i] /= pivot
    c.polys[i, :] /= pivot

    for j in range(c.basis_dimension):
        if i == j:
            continue
        coef = c.values[i, j]
        c.values[:, j] -= coef * c.values[:, i]
        c.polys[j, :] -= coef * c.polys[i, :]


def perform_lu_factorization(params: LagrangeParams) -> Certification:
    logging.debug('computing lagrange polynomials')

    c = params.construct_cert()
    _test_v(c)

    for i in range(c.basis_dimension):
        _test_v(c)

        # do not replace first point
        pivot_index = (
            i + np.argmax(np.abs(c.values[i:, i]))
            if i > 0 else 0
        )
        pivot_value = abs(c.values[pivot_index, i])
        if pivot_value < params.xsi_replace:
            point, val = _find_replacement(c, c.polys[i, :].copy(), params, '')
            assert not np.isnan(val) and abs(val) > params.xsi_replace, 'Unable to find replacement point' + str(abs(val)) + str(params.xsi_replace)
            _replace_point(c, i, point)
            _test_v(c)
            print("Replaced point at " + str(i) + " with " + str([xi for xi in point]))
        else:
            _swap_rows(c, i, pivot_index)
            _test_v(c)

        print("Row " + str(i) + ", pivot value of " + str(pivot_value))
        _reduce(c, i)
        _test_v(c)

    _test_v(c)

    # # TODO: This might not include all points...
    # n = c.current_sample.shape[1]
    # p = params.plotter.create_plot(
    #     'model_improvement',
    #     Bounds.create(-2.2 * np.ones(n), 2.2 * np.ones(n)),
    #     'sample set improvement',
    #     subfolder='lagrange')

    # original = params.sample_region.shift_sample(c.original_sample)
    # improved = c.current_sample
    # has = lambda arr, x: any([np.linalg.norm(x - v) < 1e-12 for v in arr])

    # discarded = np.array([x for x in original if not has(improved, x)])
    # kept = np.array([x for x in original if has(improved, x)])
    # added = np.array([x for x in improved if not has(original, x)])
    # if len(discarded) > 0:
    #     p.add_points(discarded, label='discarded points', color='red', marker='x')
    # if len(kept) > 0:
    #     p.add_points(kept, label='kept points', color='blue', marker='+', s=50)
    # if len(added) > 0:
    #     p.add_points(added, label='added points', color='green',  marker='o')
    # p.add_circle(np.zeros(2), 1, label='trust region')
    # p.save()

    # c.quadratics = []
    # for i in range(c.polys.shape[0]):
    #     poly = Polynomial.construct_polynomial(params.basis, c.polys[i, :])
    #     quad = poly.to_matrix_form()
    #     c.quadratics.append(quad)

    #     success, ppoint, pval = solve_tr_subproblem(quad.g, 2 * quad.Q)
    #     assert success, 'Unable to compute lambda'
    #     success, npoint, nval = solve_tr_subproblem(-quad.g, -2 * quad.Q)
    #     assert success, 'Unable to compute lambda'
    #     Lambda = max(abs(pval), abs(nval))
    #     if c.Lambda is None or Lambda > c.Lambda:
    #         c.Lambda = Lambda

    #     for idx in range(c.basis_dimension):
    #         x = c.current_sample[idx]
    #         val = poly.evaluate(x)
    #         TOL = 1e-6
    #         if i == idx:
    #             assert np.abs(1.0 - val) < TOL, 'lagrange polynomial failed to be one at its own point' \
    #                 ', found: ' + str(val)
    #         else:
    #             assert np.abs(val) < TOL, 'lagrange polynomial failed to be zero at another point' \
    #                 ', found: ' + str(val)

    #         val = quad.evaluate(x)
    #         if i == idx:
    #             assert np.abs(1.0 - val) < TOL, 'lagrange polynomial failed to be one at its own point' \
    #                 ', found: ' + str(val)
    #         else:
    #             assert np.abs(val) < TOL, 'lagrange polynomial failed to be zero at another point' \
    #                 ', found: ' + str(val)

    #     if params.plot_maximizations:
    #         name = str(poly) + ', lambda = ' + str(Lambda)
    #         p = params.plotter.create_plot(
    #             'lambda_calculation',
    #             Bounds.create([-1.2, -1.2], [1.2, 1.2]),
    #             name,
    #             subfolder='lagrange')
    #         p.add_contour(quad.evaluate, label='lagrange polynomial')
    #         p.add_points(c.current_sample, label='replacement points', color='red', marker='x')
    #         p.add_point(ppoint, label='min value = ' + str(pval), color='green', marker='o')
    #         p.add_point(npoint, label='max value = ' + str(nval), color='blue', marker='o')
    #         p.add_circle(np.zeros(2), 1, label='trust region', color='k')
    #         p.save()

    # params.logger.verbose('Lambda = ' + str(c.Lambda))
    # for idx, quad in enumerate(c.quadratics):
    #     params.logger.verbose_json('lagrange polynomial ' + str(idx), quad)
    # params.logger.stop_step()
    return c

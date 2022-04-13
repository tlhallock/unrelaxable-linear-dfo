import logging

from alg.state import State
from alg.criticality import check_criticality
from alg.geometry import improve_geometry
from alg.model import udpate_model
from alg.run_result import IterationResult, RunResult

# def begin_iteration(state):
# 	state.logger.start_step('iteration ' + str(state.iteration))
# 	state.history.add_outer_tr(iteration=state.iteration, iterate=state.current_iterate, radius=state.outer_tr_radius)
# 	state.history.add_sample_region(state.iteration, state.sample_region)

# 	state.buffering_plot = None
# 	state.current_plot = state.plotter.create_plot(
# 		'iteration_' + str(state.iteration),
# 		Bounds.create(
# 			state.current_iterate - 1.5 * state.outer_tr_radius,
# 			state.current_iterate + 1.5 * state.outer_tr_radius),
# 		'Iteration ' + str(state.iteration),
# 		subfolder='iterations'
# 	)

# 	state.current_plot.add_contour(
# 		state.sample_region.evaluate,
# 		**state.params.plot_options['sample-region'])
# 	state.current_plot.add_linf_tr(
# 		state.current_iterate, state.outer_tr_radius,
# 		**state.params.plot_options['trust-region'])


# def end_iteration(state, iteration_result, model=None, br=None):
# 	state.current_plot.add_point(
# 		state.current_iterate,
# 		**state.params.plot_options['current-iterate'])

# 	state.logger.info_json('iteration result', iteration_result)

# 	if iteration_result.completed:
# 		state.current_plot.save()
# 		return iteration_result

# 	state.unbound_radius = iteration_result.update_radius and iteration_result.radius_update > 1.0

# 	if iteration_result.infeasible_sample_region:
# 		repair_sample_region(state)
# 		state.current_plot.save()
# 		return iteration_result

# 	old_radius = state.outer_tr_radius
# 	old_center = state.current_iterate
# 	if iteration_result.update_radius:
# 		state.outer_tr_radius *= iteration_result.radius_update

# 	if iteration_result.update_iterate:
# 		state.current_iterate = iteration_result.new_iterate
# 		construct_sample_region(state, model, br)
# 	elif iteration_result.update_radius:
# 		scale_sample_region(state, model, factor=iteration_result.radius_update, old_radius=old_radius)

# 	if model is not None and state.sample_region is not None:
# 		next_polyhedron = model.get_all_unshifted_constraints()
# 		if not state.sample_region.contained_within_polyhedron(next_polyhedron, tol=1e-4):
# 			print('sample region is out of bounds')

# 	if state.buffering_plot is not None:
# 		state.buffering_plot.add_point(
# 			(state.current_iterate - old_center) / old_radius, label='next iterate', s=50, marker='o')
# 		state.buffering_plot.add_contour(
# 			lambda x: state.sample_region.evaluate(old_center + old_radius * x),
# 			lvls=[0], label='next sample region')

# 	return iteration_result


# def log_iteration(state, model, br, it_result, crit_check, cert):
# 	file_name = 'iteration_' + str(state.iteration) + '.json'
# 	dir_name = os.path.join(state.root_directory, 'iteration_json')
# 	os.makedirs(dir_name, exist_ok=True)
# 	state.logger.info('trust region: {center=' + str(state.current_iterate) + ',radius=' + str(state.outer_tr_radius) + '}')
# 	with open(os.path.join(dir_name, file_name), 'w') as json_out:
# 		json_out.write(JsonUtils.dumps({
# 			'state': state.to_json(show_history=False),
# 			'model': model,
# 			'buffered-region': br,
# 			'iteration-result': it_result,
# 			'criticality-check': crit_check,
# 			'certification': cert,
# 		}))

# 	with open(os.path.join(dir_name, 'history.json'), 'w') as json_out:
# 		json_out.write(JsonUtils.dumps({'history': state.history}))


def run_iteration(state: State) -> IterationResult:
    logging.info('beginning iteration', state.iteration)
    cert = improve_geometry(state)
    model = udpate_model(state, cert)
    if not model.success:
        # it_result = IterationResult.repair_sample_region()
        return IterationResult(
            success=False,
            converged=False,
            completed=False,
            message="Infeasible sample point",
        )

    criticality = check_criticality(state, model)
    if criticality.converged:
        return IterationResult(
            success=True,
            converged=True,
            completed=True,
            message="Converged",
        )
    elif criticality.critical:
    #  .multiply_radius(state.params.tr_update_dec)
        return IterationResult(
            success=True,
            converged=False,
            completed=False,
            message="Critical",
        )

# 		buffered_region = compute_buffering_cones(state, model)
# 		if not buffered_region.regular:
# 			if state.outer_tr_radius < state.params.threshold_tr_radius:
# 				it_result = IterationResult.create('irregular').set_completed()
# 				return end_iteration(state, it_result)
# 			else:
# 				it_result = IterationResult.create('empty-buffered-region').multiply_radius(state.params.tr_update_dec)
# 				return end_iteration(state, it_result)

# 		tr_solution = solve_tr_subproblem(state, model, buffered_region)
# 		it_result = test_for_improvement(state, model, tr_solution)
# 		return end_iteration(state, it_result, model, buffered_region)

# 		# TODO: set the next bounded_radius....
# 	finally:
# 		log_iteration(state, model, buffered_region, it_result, criticality_check, certification)
# 		state.current_plot.save()
# 		if state.buffering_plot is not None:
# 			state.buffering_plot.save()
# 		state.history.create_plot(
# 			state.plotter,
# 			iterations=(state.iteration - 50, state.iteration),
# 			subfolder='histories')
# 		state.logger.stop_step()
    return IterationResult(
        success=False,
        converged=False,
        completed=False,
        message="unreachable",
    )


def run_algorithm(state: State) -> RunResult:
    logging.info('Running algorithm', state.problem.name)
 
    while state.iteration < state.params.maximum_iterations:
        state.iteration += 1
        it_result = run_iteration(state)
        if it_result.completed:
            return RunResult.from_iteration(it_result)

    return RunResult(
        success=False,
        converged=False,
        message="Hit maximum iterations"
    )

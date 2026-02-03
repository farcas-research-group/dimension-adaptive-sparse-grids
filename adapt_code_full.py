from sg_lib.grid.grid import *
from sg_lib.algebraic.multiindex import *
from sg_lib.operation.interpolation_to_spectral import *
from sg_lib.adaptivity.spectral_scores import *

# high-fidelity model for this test
def hi_fi_model(x):
	test = np.cos(np.pi + 1.0*x[0] + 0.55*x[1] + 0.8*x[2] + 0.1*x[3]) + 1.0

	return test
    
if __name__ == '__main__':

	# number of parameters
	dim = 4

	# sparse grid setup
	# here, we consider a uniform input distribution, thus the bounds are [0, 1]^dim
	left_bounds 	= np.zeros(dim)
	right_bounds 	= np.ones(dim)
	# sparse grids always begin at level 1
	grid_level 		= 1
	# not important; keep it 1 for interpolation
	level_to_nodes 	= 1
	# probability density function; in this case, we have a uniform density in [0, 1]^dim
	# since the stochastic inputs are independent, it has a product structure
	weights 		= [lambda x: 1. for i in range(dim)]

	# tolerance used in the adaptive algorithm
	tols 		= 1e-6*np.ones(dim + 1)
	# maximum level reachable by the sparse grid
	max_level 	= 20

	# first multiindex in K is (1, 1, ... , 1)
	init_multiindex = np.ones(dim, dtype=int)
	
	# create objects to do sensitivity-driven adaptve sparse grid interpolation

	# grid object
	Grid_obj 				= Grid(dim, grid_level, level_to_nodes, left_bounds, right_bounds, weights)
	# multiindex object
	Multiindex_obj 			= Multiindex(dim)

	# interpolation object
	InterpToSpectral_obj 	= InterpolationToSpectral(dim, level_to_nodes, left_bounds, right_bounds, weights, max_level, Grid_obj)

	# adaptivity object; the most important thing is the refinement indicator; see the paper
	# see also the implementation
	Adaptivity_obj 			= SpectralScores(dim, tols, init_multiindex, max_level, level_to_nodes, InterpToSpectral_obj)

	# add initial multiindex to the multiindex set, aka, K = {(1, 1, ..., 1)}
	init_multiindex_set = Multiindex_obj.get_std_total_degree_mindex(grid_level)
	# take the grid points corresponding to the first multiindex
	init_grid_points 	= Grid_obj.get_std_sg_surplus_points(init_multiindex_set)
	init_no_points 		= Grid_obj.get_no_fg_grid_points(init_multiindex_set)

	InterpToSpectral_obj.get_local_global_basis(Adaptivity_obj)

	# begin the adaptive process

	# first step, do the initial subspace which contains 1 point
	for sg_point in init_grid_points:
		sg_val = hi_fi_model(sg_point)
		InterpToSpectral_obj.update_sg_evals_all_lut(sg_point, sg_val)

	InterpToSpectral_obj.update_sg_evals_multiindex_lut(init_multiindex, Grid_obj)
	
	# adaptivity begins here; see paper, especially the algorithms, for more details
	Adaptivity_obj.init_adaption()

	prev_len 		= len(init_no_points)
	no_adapt_steps 	= 0
	total_len 		= 1
	while not Adaptivity_obj.stop_adaption:
		no_adapt_steps += 1

		new_multiindices = Adaptivity_obj.do_one_adaption_step_preproc()

		for multiindex in new_multiindices:
			new_grid_points = Grid_obj.get_sg_surplus_points_multiindex(multiindex)
			total_len 		+= len(new_grid_points)

			for sg_point in new_grid_points:
				sg_val = hi_fi_model(sg_point)
			
				InterpToSpectral_obj.update_sg_evals_all_lut(sg_point, sg_val)

			InterpToSpectral_obj.update_sg_evals_multiindex_lut(multiindex, Grid_obj)
			
		Adaptivity_obj.do_one_adaption_step_postproc(new_multiindices)
		Adaptivity_obj.check_termination_criterion()

	print('adaptivity done after {} steps'.format(no_adapt_steps))
	print('total grid size: {} grid points'.format(total_len))

	InterpToSpectral_obj.get_local_global_basis(Adaptivity_obj)

	# now we want to be able to query the obtained surrogate model
	adapt_sg_surrogate = lambda x: InterpToSpectral_obj.eval_operation_sg(Adaptivity_obj.multiindex_set, x)	

	# here, we compute the Pearson correlation coefficient between the high- and low-fidelity model, which will be relevant for doing  MFMC
	np.random.seed(9812788)
	no_test_samples 	= 20
	test_samples 		= np.random.uniform(0, 1, size=(no_test_samples, dim))

	print('testing the approximation accuracy using {} testing samples'.format(no_test_samples))
	print('*' * 80)
	for sample in test_samples:
		print('reference result: {:.4} vs surrogate approximation: {:.4}'.format(hi_fi_model(sample), adapt_sg_surrogate(sample)))
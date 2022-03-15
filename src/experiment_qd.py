from qdpy import algorithms, containers, benchmarks, plots
from qdpy.algorithms import *
from qdpy.containers import *
from qdpy.benchmarks import *
from qdpy.plots import *
from qdpy.base import *
from qdpy import tools
from racecar import Controller, Racecar
from arm import Arm
from racecar import interpolate_keypoints, interpolate_motors
from environment import Environment

import numpy as np



def main():
	if len(sys.argv) > 2:
		nb_iterations = int(sys.argv[1])
		grid_size = int(sys.argv[2])
		batch_size = int(sys.argv[3])
		arm_dimension = int(sys.argv[4])
		controller_center = (float(sys.argv[5]), float(sys.argv[6]))
		controller_radius = float(sys.argv[7])
	else:
		nb_iterations = 2
		grid_size = 64
		batch_size = 100
		arm_dimension = 4
		controller_center = (0.6, 0.5)
		controller_radius = 0.1

	print("nb_iterations:" + str(nb_iterations))
	print("grid_size:" + str(grid_size))
	print("batch_size:" + str(batch_size))
	print("arm_dimension:" + str(arm_dimension))
	print("controller_center:" + str(controller_center))
	print("controller_radius:" + str(controller_radius))

	arm_positions = 10
	total_dim = arm_dimension * arm_positions

	env = Environment(arm_dimension=arm_dimension, controller_center=controller_center, controller_radius=controller_radius, keypoint_spacing=5)
	# arm_configs = []
	# for _ in range(400):
	# 	arm_configs.append((np.random.rand() * 2 - 1) * np.pi/3)
	# # print(arm_configs)
	# print(env.update(arm_configs))

	def _illuminate_artificial_landscape(ind, func, nb_features = 2):
		fitness = func(ind)
		features = env.update(ind).tolist()
		return fitness, features

	def zero_fitness(ind):
		return (0,)

	############################################################################
	# Map Elites
	############################################################################

	ind_domain = (-np.pi/3., np.pi/3.)
	nb_features = 2

	grid = containers.Grid(
				shape=(grid_size, grid_size),
				max_items_per_bin=1,
				fitness_domain=((0., 0.),),
				features_domain=[(70, 90), (-80, -60)],
				storage_type=list,
			  )

	grid_algo = algorithms.RandomSearchMutPolyBounded(
			  grid,
			  budget=batch_size * nb_iterations,
			  batch_size=batch_size,
			  dimension=total_dim,
			  optimisation_task="maximisation",
			  ind_domain=ind_domain,
			  sel_pb=0.5, 
			  init_pb=0.5,
			  mut_pb=0.4,
			  eta=20.0,
			  name="grid_algo"
			)

	# Create a logger to pretty-print everything and generate output data files
	# grid_evaluator = AlgoEvaluation(grid_algo, evaluate_reaching_att, qd_coverage, testcases, evaluate_iters)
	grid_logger = algorithms.AlgorithmLogger(grid_algo)

	illu_fitness = partial(_illuminate_artificial_landscape, func=zero_fitness)

	# Define evaluation function
	grid_eval_fn = partial(algorithms.partial(illu_fitness, nb_features=nb_features))

	# Run illumination process !
	best = grid_algo.optimise(grid_eval_fn)

if __name__ == '__main__':
	main()
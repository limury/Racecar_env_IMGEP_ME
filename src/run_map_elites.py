import sys
from environment import Environment, GRBFTrajectory
import numpy as np
import time
from pymap_elites.map_elites import cvt as cvt_map_elites
from pymap_elites.map_elites import common as cm




gui = False # Plot environment animation ?
n = 10000 # Number of iterations

t = time.time()

stick1_moved = 0
stick2_moved = 0
obj1_moved = 0
obj2_moved = 0


# storage directory for results
log_dir = sys.argv[1]
# num evals
n_evals = int(sys.argv[2])
# batch size
b_size = int(sys.argv[3])
# dump period
dump_period = int(sys.argv[4])
# selector to use [ "uniform", "curiosity", "curiosity-child" ]
selector = sys.argv[5]
# mutation operator to use [ "iso_dd", "polynomial", "sbx" ]
mutation = sys.argv[6]
# additional mutations
variant = "" 
if len(sys.argv) > 7:
    variant = sys.argv[7:]

nb_joints = 4
nb_waypoints = 5
grid_sizes = [2, 50, 50]
D_Pad_position = (0.6, 0.5)
D_Pad_radius = 0.2
steps_per_basis=10

arm_env_bounds = [ \
    [80, 100], # bounds for x coord
    [-80, -60], # bounds for y coord
]

offset_x = arm_env_bounds[0][0]
scale_x = arm_env_bounds[0][1] - arm_env_bounds[0][0]
offset_y = arm_env_bounds[1][0]
scale_y = arm_env_bounds[1][1] - arm_env_bounds[1][0]

# returns tuple of fitness and BD
def eval( geno ):
    geno = (geno * 2 - 1) * np.pi
    env = Environment(  nb_joints=nb_joints, 
                        controller_center=D_Pad_position,
                        controller_radius=D_Pad_radius,
                        steps_per_basis=steps_per_basis )

    BD = env.update( geno, rbf_generator=traj_generator)
    
    BD[0] = (BD[0] + 1) / 2
    BD[1] = (BD[1] + 1) / 2
    BD[2] = (BD[2] - offset_x) / scale_x
    BD[3] = (BD[3] - offset_y) / scale_y

    return 0, [ (0.25, BD[0], BD[1]), (0.75, BD[2], BD[3]) ]
    


if __name__ == "__main__":
    traj_generator = GRBFTrajectory(n_dims=nb_joints, 
                                   sigma=steps_per_basis//2,
                                   steps_per_basis=steps_per_basis,
                                   max_basis=nb_waypoints)
    px = \
    {
        # can be: curiosity, BD, centroid, fitness, genotype
        "log_outputs": ['centroid', 'curiosity'],
        # more of this -> higher-quality CVT
        "cvt_samples": 25000,
        # we evaluate in batches to parallelize
        "batch_size": b_size,
        # proportion of niches to be filled before starting
        "random_init": 0.002,
        # batch for random initialization
        "random_init_batch": 200,
        # when to write results (one generation = one batch)
        "dump_period": dump_period,
        # do we use several cores?
        "parallel": True,
        # do we cache the result of CVT and reuse?
        "cvt_use_cache": False,
        # min/max of parameters of genotype
        "min": 0,
        "max": 1,
        # probability of mutating each number in the genotype
        "mutation_prob": 0.4,
        # selector ["uniform", "curiosity", "random_search"]
        "selector" : selector,
        # mutation operator ["iso_dd", "polynomial", "sbx", "gaussian", "curiosity_func"]
        "mutation" : mutation,
        "variant" : variant, 
        "eta_m" : 5.0,
        "iso_sigma": 0.1,
        "line_sigma": 0.2,
        "std_dev" : 0.02
        }

    functions = []
    if 'curiosity_func' in px['mutation']:
        functions = [ cm.iso_dd, cm.sbx, cm.polynomial_mutation ]
    else:
        if 'iso_dd' in px['mutation']:
            functions.append( cm.iso_dd )
        if 'sbx' in px['mutation']:
            functions.append( cm.sbx )
        if 'polynomial' in px['mutation']:
            functions.append( cm.polynomial_mutation )
    



    BD_size = 1
    for x in grid_sizes:
        BD_size *= x

    archive = cvt_map_elites.MapElites( dim_map=3, 
                                        dim_x=nb_joints*nb_waypoints, 
                                        eval_func=eval, 
                                        n_niches=BD_size, 
                                        max_evals=n_evals, 
                                        log_file=None, 
                                        mutation_func=functions,
                                        params=px, 
                                        bins=grid_sizes, 
                                        log_dir=log_dir)
    


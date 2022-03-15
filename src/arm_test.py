#from explauto.environment import environments
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from environment import Environment, GRBFTrajectory

from explauto.experiment import make_settings
from explauto import SensorimotorModel, InterestModel, Agent, Experiment

nb_joints = 4
nb_waypoints = 10
grid_sizes = [2, 100, 100]
D_Pad_position = (0.6, 0.5)
D_Pad_radius = 0.1
steps_per_basis=5

def main():
    traj_generator = GRBFTrajectory(n_dims=nb_joints, 
                                   sigma=steps_per_basis//2,
                                   steps_per_basis=steps_per_basis,
                                   max_basis=nb_waypoints)
    env = Environment(  nb_joints=nb_joints, 
                        controller_center=D_Pad_position,
                        controller_radius=D_Pad_radius,
                        steps_per_basis=steps_per_basis )
    
    individual = np.array([np.pi, 0, np.pi/2, -np.pi/2, 0, 0, 0, 0])
    print('individual: ', individual)
    # generate trajectory using Radial Basis Function
    arm_commands = traj_generator.trajectory(individual)*np.pi
    print('arm_commands: ', arm_commands)
    # get trajectory performed by the robot arm
    arm_trajectory = np.array([env.arm.env.update(angles) for angles in arm_commands])
    print("coordinates of end affector: ", arm_trajectory)

    fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(25, 5))


    for index, motor in enumerate(arm_commands):
        env.arm.env.plot_arm(axes[index], arm_commands[index])
        axes[index].set_xlim([-1, 1])
        axes[index].set_ylim([-1, 1])
    plt.savefig('uhm.png')
    

if __name__ == '__main__':
    main()

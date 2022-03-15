#from explauto.environment import environments
from matplotlib import pyplot as plt
import numpy as np
from environment import Environment, GRBFTrajectory

from explauto.experiment import make_settings
from explauto import SensorimotorModel, InterestModel, Agent, Experiment

nb_joints = 4
nb_waypoints = 10
grid_sizes = [2, 100, 100]
D_Pad_position = (0.715, 0.70)
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
    
    individual = np.array([np.pi/4, 0, 0, 0]*5)
    print('individual: ', individual)
    # generate trajectory using Radial Basis Function
    arm_commands = traj_generator.trajectory(individual)*np.pi
    arm_commands = np.array([[np.pi/4, 0, 0, 0]]*40)
    print('arm_commands: ', arm_commands)
    # get trajectory performed by the robot arm
    arm_trajectory = np.array([env.arm.env.update(angles) for angles in arm_commands])
    print("coordinates of end affector: ", arm_trajectory)

    actions = [env.car_controller.get_action(point) for point in arm_trajectory]
    actions =  [[-1,1,0]] * 10 + [[0,1,0]] * 60
    print("car actions: ",actions)
    # simulate racecar using D-Pad controls
    final_car_position = env.racecar.simulate(actions, render=True)
    print("final car pos: ", final_car_position)

    #fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(25, 5))


    #for index, motor in enumerate(arm_commands):
    #    env.arm.env.plot_arm(axes[index], arm_commands[index])
    #    axes[index].set_xlim([-1, 1])
    #    axes[index].set_ylim([-1, 1])
    #plt.show(block=True)
    

if __name__ == '__main__':
    main()

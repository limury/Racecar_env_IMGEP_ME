from racecar import Controller, Racecar
from arm import Arm
from racecar import interpolate_keypoints, interpolate_motors
from box2d_racecar import CarRacing
import numpy as np

class GRBFTrajectory(object):
    # n_dims = nb_joints
    def __init__(self, n_dims, sigma, steps_per_basis, max_basis):
        self.n_dims = n_dims
        self.sigma = sigma
        self.alpha = - 1. / (2. * self.sigma ** 2.)
        self.steps_per_basis = steps_per_basis
        self.max_basis = max_basis
        self.precomputed_gaussian = np.zeros(2 * self.max_basis * self.steps_per_basis)
        for i in range(2 * self.max_basis * self.steps_per_basis):
            self.precomputed_gaussian[i] = self.gaussian(self.max_basis * self.steps_per_basis, i)
        
    def gaussian(self, center, t):
        return np.exp(self.alpha * (center - t) ** 2.)
    
    def trajectory(self, weights):
        n_basis = len(weights)//self.n_dims
        weights = np.reshape(weights, (n_basis, self.n_dims)).T
        steps = self.steps_per_basis * n_basis
        # trajectory has dimensions: nb_steps * nb_joints
        traj = np.zeros((steps, self.n_dims))
        for step in range(steps):
            g = self.precomputed_gaussian[self.max_basis * self.steps_per_basis + self.steps_per_basis - 1 - step::self.steps_per_basis][:n_basis]
            traj[step] = np.dot(weights, g)
        return np.clip(traj, -1., 1.)
    
    def plot(self, traj):
        plt.plot(traj)
        plt.ylim([-1.05, 1.05])


class Environment():
    def __init__(self, nb_joints, controller_center, controller_radius, steps_per_basis):
        self.nb_joints = nb_joints
        # initialize the explauto simple arm
        self.arm = Arm(nb_joints=nb_joints)
        # initialize the D_Pad the arm will control
        self.car_controller = Controller(controller_center, controller_radius)
        # initialize the racecar the D_pad will control
        self.racecar = Racecar()
        self.steps_per_basis = steps_per_basis

	# individual = genotype
    # genotype has X arm poisitons for each of Y joints so geno size = X * Y
    def update(self, individual: np.ndarray, rbf_generator: GRBFTrajectory):
        # generate trajectory using Radial Basis Function
        arm_commands = rbf_generator.trajectory(individual)*np.pi
        # get trajectory performed by the robot arm
        arm_trajectory = np.array([self.arm.env.update(angles) for angles in arm_commands])
        # check what the arm has pressed on the D-Pad
        actions=[]
        for point in arm_trajectory:
            action = self.car_controller.get_action(point)
            actions.extend([action, action, action])

        # simulate racecar using D-Pad controls
        final_car_position = self.racecar.simulate(actions, render=False)
        # return final position of the arm and final position of the car
        return np.concatenate( (arm_trajectory[-1], final_car_position), axis=None )

    def plot_trajectory(self, individual, ax, rbf_generator: GRBFTrajectory):
        # generate trajectory using Radial Basis Function
        arm_commands = rbf_generator.trajectory(individual)*np.pi
        # get trajectory performed by the robot arm
        arm_trajectory = np.array([self.arm.env.update(angles) for angles in arm_commands])

        points_x = [point[0] for point in arm_trajectory]
        points_y = [point[1] for point in arm_trajectory]

        ax.plot(points_x, points_y)

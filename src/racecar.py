import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.animation import FuncAnimation
from functools import partial
import gym
from arm import Arm
from box2d_racecar import CarRacing

import time

class Racecar():
    # import car racing env from box2d
    # clear environment
    def __init__(self):
        self.env = CarRacing(verbose=0)
    
    # actions = array of directions
    def simulate(self, actions, render=False):
        # copy pasting this code so the render value is only checked once
        self.env.reset()
        if (render):
            for action in actions:
                self.env.render()
                observation, reward, done, info = self.env.step(action)
                time.sleep(0.1)
        else:
            for action in actions:
                observation, reward, done, info = self.env.step(action)
        self.env.close()
        return self.env.car.hull.position
        

class Controller():
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
    def check_region(self, point):
        dist_to_center = np.linalg.norm(point - self.center)
        region = -1
        if dist_to_center <= self.radius:
            dy = point[1] - self.center[1]
            dx = point[0] - self.center[0]
            angle = np.arctan2(dy, dx)
            region = int(np.floor((angle + np.pi/8) / np.pi * 4)) % 8
        return region
    def get_action(self, point):
        region = self.check_region(point)
        steps = [[1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0], [-1, 0, 0], [-1, 0, 0.8], [0, 0, 0.8], [1, 0, 0.8]]
        return steps[region]
    def draw(self, ax):
        circle = plt.Circle(self.center, self.radius)
        long_side = self.radius * np.cos(np.pi/8)
        short_side = self.radius * np.sin(np.pi/8)
        line1 = ConnectionPatch(xyA=(self.center[0] + long_side, self.center[1] + short_side), 
            xyB=(self.center[0] - long_side, self.center[1] - short_side),
            coordsA='data',
            coordsB='data',
            lw=1)
        line2 = ConnectionPatch(xyA=(self.center[0] + long_side, self.center[1] - short_side), 
            xyB=(self.center[0] - long_side, self.center[1] + short_side),
            coordsA='data',
            coordsB='data',
            lw=1)
        line3 = ConnectionPatch(xyA=(self.center[0] + short_side, self.center[1] + long_side), 
            xyB=(self.center[0] - short_side, self.center[1] - long_side),
            coordsA='data',
            coordsB='data',
            lw=1)
        line4 = ConnectionPatch(xyA=(self.center[0] - short_side, self.center[1] + long_side), 
            xyB=(self.center[0] + short_side, self.center[1] - long_side),
            coordsA='data',
            coordsB='data',
            lw=1)
        ax.add_artist(circle)
        ax.add_artist(line1)
        ax.add_artist(line2)
        ax.add_artist(line3)
        ax.add_artist(line4)
        
def generate_keypoints(n):
    keypoints = []
    for _ in range(n):
        keypoints.append(np.random.rand(2) * 2 - (1, 1))
    return keypoints

def interpolate_keypoints(initpos, keypoints, space):
    first = True
    prev = initpos
    new_list = []
    for keypoint in keypoints:
        if first:
            new_list = np.linspace(prev, keypoint, num=space)
            first = False
        else:
            new_list = np.concatenate((new_list, np.linspace(prev, keypoint, num=space)), axis=0)
        prev = keypoint
    return new_list

def interpolate_motors(initconfig, motor_configs, space):
    first = True
    prev = initconfig
    new_list = []
    for config in motor_configs:
        if first:
            new_list = np.linspace(prev, config, num=space)
            first = False
        else:
            new_list = np.concatenate((new_list, np.linspace(prev, config, num=space)), axis=0)
        prev = config
    return new_list

def joint_positions(angles, lengths, unit='rad'):
    """ Link object as defined by the standard DH representation.
    :param list angles: angles of each joint
    :param list lengths: length of each segment
    :returns: x positions of each joint, y positions of each joints, except the first one wich is fixed at (0, 0)
    .. warning:: angles and lengths should be the same size.
    """
    if len(angles) != len(lengths):
        raise ValueError('angles and lengths must be the same size!')

    if unit == 'rad':
        a = np.array(angles)
    elif unit == 'std':
        a = np.pi * np.array(angles)
    else:
        raise NotImplementedError
     
    a = np.cumsum(a)
    return np.cumsum(np.cos(a)*lengths), np.cumsum(np.sin(a)*lengths)

def lengths(n_dofs, ratio):
    l = np.ones(n_dofs)
    for i in range(1, n_dofs):
        l[i] = l[i-1] / ratio
    return l / sum(l)




def main():
    arm = Arm(arm_dimension=4)
    controller = Controller((0.5, 0.6), 0.3)
    env = CarRacing()
    observation = env.reset()

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    controller.draw(ax)
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'o-', lw=2, mew=2, color='black')

    def init():
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        return ln,

    def update(frame):
        env.render()
        pos = arm.environment.update(frame)
        x, y = joint_positions(frame, lengths(4, 1.5))
        x = np.hstack((0., x))
        y = np.hstack((0., y))
        action = controller.get_action(pos)
        observation, reward, done, info = env.step(action)
        
        if done:
            observation = env.reset()
        ln.set_data(x, y)
        return ln,

    configs = interpolate_motors([0] * 4, arm.environment.random_motors(n=100), 5)
    points = [arm.environment.update(pos) for pos in configs]
    # print(points)
    # points = interpolate_keypoints((0, 0), generate_keypoints(100), 20)
    ani = FuncAnimation(fig, update, frames=configs,
                        init_func=init, blit=False, repeat=True, interval=100)
    env.close()
    plt.show(block=True)
    plt.close()
    

def test():
    controller = Controller((0.3, 0.3), 0.25)
    print(controller.check_region(np.array([0.35, 0.1])))

if __name__ == '__main__':
    main()
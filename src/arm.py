from explauto import Environment
from explauto.environment import environments
from matplotlib import pyplot as plt
import numpy as np

from explauto.experiment import make_settings
from explauto import SensorimotorModel, InterestModel, Agent, Experiment

class Arm():
    def __init__(self,  nb_joints):
        #random_goal_babbling = make_settings(environment='simple_arm', environment_config = 'mid_dimensional',
        #                                     babbling_mode='goal', 
        #                                     interest_model='random',
        #                                     sensorimotor_model='nearest_neighbor')

        env_cls, _, _ = environments['simple_arm']

        #########################################################################
        # Experiment params
        #########################################################################

        BD_min = -1.
        BD_max = 1.

        #########################################################################
        # Plotting params
        #########################################################################

        y_min = 0
        y_max = 0.4

        arm_config = {
            'length_ratio': 1.5,
            'm_maxs': [np.pi] *  nb_joints,
            'm_mins': [-np.pi] *  nb_joints,
            'noise': 0.02,
            's_maxs': [BD_min, BD_max],
            's_mins': [BD_min, BD_max]
        }
        self.env = env_cls(**arm_config)

def main():
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
    arm = Arm(4)

    motor_configurations = arm.env.random_motors(n=2)
    new_list = np.linspace(motor_configurations[0], motor_configurations[1], num=5)
    print(new_list)

    for index, motor in enumerate(new_list):
        arm.env.plot_arm(axes[index], new_list[index])
        axes[index].set_xlim([-1, 1])
        axes[index].set_ylim([-1, 1])
    plt.show(block=True)
    

if __name__ == '__main__':
    main()

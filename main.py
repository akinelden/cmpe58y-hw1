import gym
import numpy as np
import matplotlib.pyplot as plt
from QLearning import QLearner

env = gym.make('CartPole-v0')
learner = QLearner(env, 20, space_limits=[
                    (-3, 3), (-5, 5), (-0.4, 0.4), (-5, 5)], alpha=0.15, gama=0.95, epsilon=0.99, decay_rate=0.99995)
learner.run(1000, render=False)
learner.run(20, render=True, log=True)
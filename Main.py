# Simple implementation of usage MDP Toolbox library

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import mdptoolbox as mdp

print(">>>AI Project 4_1<<<")

world = mdp.LoadGridWord('MDPRL_world0.data', 1)

print(">>>VALE_ITERATION runs for {} iterations, GAMMA: {}<<<".format(iter, world.Gamma))
iter = 200
V, A = mdp.vi.ValueIteration(world, iter)
mdp.PrintPolicyVal(world, V, A)

print(">>>Q-LEARNING runs for {} iterations... be patient :D <<<".format(iter))
iter = 100000
Q = mdp.ql.QLearning(world, iter)
mdp.PrintQResults(world, Q)

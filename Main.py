# Simple implementation of usage MDP Toolbox library

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import mdptoolbox as mdp

print(">>>AI Project 4_1<<<")

world = mdp.LoadGridWord('MDPRL_world0.data', 0)

iter = 200
print(">>>VALE_ITERATION runs for {} iterations, GAMMA: {}<<<".format(iter, world.Gamma))
V, A = mdp.vi.ValueIteration(world, iter)
mdp.PrintPolicyVal(world, V, A)

world = mdp.LoadGridWord('MDPRL_world1.data', 0)

iter = 200
print(">>>VALE_ITERATION runs for {} iterations, GAMMA: {}<<<".format(iter, world.Gamma))
V, A = mdp.vi.ValueIteration(world, iter)
mdp.PrintPolicyVal(world, V, A)

world = mdp.LoadGridWord('MDPRL_world0.data', 0)

iter = 10000000
print(">>>Q-LEARNING runs for {} iterations... be patient :D <<<".format(iter))
Q = mdp.ql.QLearning(world, iter)
mdp.PrintQResults(world, Q)

world = mdp.LoadGridWord('MDPRL_world1.data', 0)

iter = 10000000
print(">>>Q-LEARNING runs for {} iterations... be patient :D <<<".format(iter))
Q = mdp.ql.QLearning(world, iter)
mdp.PrintQResults(world, Q)
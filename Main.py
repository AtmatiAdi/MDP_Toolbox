# Simple implementation of usage MDP Toolbox library

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import mdptoolbox as mdp

plt.close('all')

world = mdp.LoadGridWord('MDPRL_world0.data', 0)

iter = 200
print(">>>VALE_ITERATION runs for {} iterations, GAMMA: {}<<<".format(iter, world.Gamma))
V, A, tail = mdp.vi.ValueIteration(world, iter)
mdp.PrintPolicyVal(world, V, A)
mdp.ShowVal(world, tail, 'Value_Iteration_MDPRL_world0.data')


world = mdp.LoadGridWord('MDPRL_world1.data', 0)

iter = 200
print(">>>VALE_ITERATION runs for {} iterations, GAMMA: {}<<<".format(iter, world.Gamma))
V, A, tail = mdp.vi.ValueIteration(world, iter)
mdp.PrintPolicyVal(world, V, A)
mdp.ShowVal(world, tail, 'Value_Iteration_MDPRL_world1.data')

world = mdp.LoadGridWord('MDPRL_world0.data', 0)

iter = 10000
print(">>>Q-LEARNING runs for {} iterations... be patient :D <<<".format(iter))
Q, tail = mdp.ql.QLearning(world, iter)
mdp.PrintQResults(world, Q)
mdp.ShowVal(world, tail, 'Q-Learning_MDPRL_world0.data')

world = mdp.LoadGridWord('MDPRL_world1.data', 0)

iter = 10000
print(">>>Q-LEARNING runs for {} iterations... be patient :D <<<".format(iter))
Q, tail= mdp.ql.QLearning(world, iter)
mdp.PrintQResults(world, Q)
mdp.ShowVal(world, tail, 'Q-Learning_MDPRL_world1.data')
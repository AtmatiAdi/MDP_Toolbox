# Python library for solving Markov Decision problems
# Created by AtmatiAdi 29.05.2021

import numpy as np
import matplotlib.pyplot as plt
import world as World
import valueiter as vi
import qlearning as ql


def LoadGridWord(fName, loud):
    # Initalization
    world = World.Grid()
    Terminal = []
    Bspecial = []
    Forbidden = []
    ActProb = [0,0,0,0]
    width = 0
    height = 0
    Reward = 0
    P = 0
    R = 0
    # Loading and pharsing file 
    file = open(fName, 'r')
    Lines = file.readlines()
    file.close()
    if loud: print("+++Loading word: {}".format(fName))
    for line in Lines:
        arg = line.split(' ')
        if arg[0] == 'W':
            world.Width = width = int(arg[1])
            world.Height = height = int(arg[2])
            world.Size = width*height
            if loud: print("World size is: [Width: {} Height: {}]".format(width, height))
        elif arg[0] == 'S':
            world.Start = (int(arg[1])-1) + (int(arg[2])-1) * width
            if loud: print("Start state is: {}".format(world.Start))
        elif arg[0] == 'P':
            ActProb[0] = float(arg[1])
            ActProb[1] = float(arg[2])
            ActProb[2] = float(arg[3])
            ActProb[3] = 1 - (ActProb[0]+ActProb[1]+ActProb[2])
            if loud: print("Probability for actions is: {}, {}, {}, {}".format(ActProb[0],ActProb[1],ActProb[2],ActProb[3]))
        elif arg[0] == 'R':
            Reward = float(arg[1])
            if loud: print("Reward is: {}".format(Reward))
        elif arg[0] == 'G':
            world.Gamma = float(arg[1])
            if loud: print("Discount factor is: {}".format(world.Gamma))
        elif arg[0] == 'E':
            world.Explor = float(arg[1])
            if loud: print("Exploration factor is: {}".format(world.Explor))
        elif arg[0] == 'T':
            state = (int(arg[1])-1) + (int(arg[2])-1) * width
            reward = float(arg[3])
            Terminal.append([state,reward])
            if loud: print("Terminal state: {}, with reward: {}".format(state, reward))
        elif arg[0] == 'B':
            state = (int(arg[1])-1) + (int(arg[2])-1) * width
            reward = float(arg[3])
            Bspecial.append([state,reward])
            if loud: print("Special state: {}, with reward: {}".format(state, reward))
        elif arg[0] == 'F':
            state = (int(arg[1])-1) + (int(arg[2])-1) * width
            Forbidden.append(state)
            if loud: print("Forbidden state: {}".format(state))
    if world.Size == 0 or sum(ActProb) != 1 or len(Terminal) == 0:
        print("World file is not complete !!")
    # Building P matrix and R vector
    P = np.zeros(shape=(len(ActProb), world.Size, world.Size))
    for act in range(len(P)):
        for state in range(len(P[1])):
            # AT FIRST CHECK IF STATE IS NOT TERMINAL OR FORBIDDEN
            found  = 0
            for tState in Terminal:
                if state == tState[0]:
                    found = 1
            for fState in Forbidden:
                if state == fState:         # forbidden state behaves like endless pit
                    found = 1               
            if found == 1:
                P[act,state,state] = 1
                continue
            # CALUCLATE PRIM STATES
            numInRow = state % width
            # MOVE LEFT
            if (numInRow - 1) >= 0:
                leftState = state - 1           # we can safetly move to left, no wall
                for fState in Forbidden:        # but there can be forbidden state
                    if leftState == fState:
                        leftState = state
            else:
                leftState = state               # bump from wall, we stay in place
            # MOVE RIGHT
            if (numInRow + 1) < width:
                rightState = state + 1          # we can safetly move to right, no wall
                for fState in Forbidden:        # but there can be forbidden state
                    if rightState == fState:
                        rightState = state
            else: 
                rightState = state              # bump from wall, we stay in place
            numInCol = state / width
             # MOVE UP
            if (numInCol + 1) < height:
                upState = state + width
                for fState in Forbidden:        # but there can be forbidden state
                    if upState == fState:
                        upState = state
            else:
                upState = state
             # MOVE DOWN
            if (numInCol - 1) >= 0:
                downState = state - width
                for fState in Forbidden:        # but there can be forbidden state
                    if downState == fState:
                        downState = state
            else: 
                downState = state
            # ASSIGN PROBABILITIES TO RIGHT STATES
            # LEFT
            if act == 0:
                P[act,state,leftState] += ActProb[0]
                P[act,state,downState] += ActProb[1]
                P[act,state,upState] += ActProb[2]
                P[act,state,rightState] += ActProb[3]
            # RIGHT
            elif act == 1:
                P[act,state,rightState] += ActProb[0]
                P[act,state,upState] += ActProb[1]
                P[act,state,downState] += ActProb[2]
                P[act,state,leftState] += ActProb[3]
            # UP
            elif act == 2:
                P[act,state,upState] += ActProb[0]
                P[act,state,leftState] += ActProb[1]
                P[act,state,rightState] += ActProb[2]
                P[act,state,downState] += ActProb[3]
            # DOWN
            elif act == 3:
                P[act,state,downState] += ActProb[0]
                P[act,state,rightState] += ActProb[1]
                P[act,state,leftState] += ActProb[2]
                P[act,state,upState] += ActProb[3]
    
    R = np.zeros(world.Size)
    for state in range(len(R)):
        R[state] = Reward
        for tState in Terminal:
            if state == tState[0]:
                R[state] = tState[1]
        for sState in Bspecial:
            if state == sState[0]:
                R[state] = sState[1]
        for fState in Forbidden:
            if state == fState:
                R[state] = 0
    if loud: 
        print("P matrix:")
        print("LEFT")
        print(P[0])
        print("RIGHT")
        print(P[1])
        print("UP")
        print(P[2])
        print("DOWN")
        print(P[3])
        print(R)
    world.P = P
    world.R = R
    return world

def PrintPolicyVal(world, Val, A):
    for row in reversed(range(world.Height)):
        rowStr = '| '
        for col in range(world.Width):
            state = world.Width * row + col
            if Val[state] >= 0:
                rowStr += ' {:1.3f} '.format(round(Val[state],3))
            else:
                rowStr += '{:1.3f} '.format(round(Val[state],3))
        rowStr += '|'
        for col in range(world.Width):
            state = world.Width * row + col
            a = A[state]
            if world.P[0,state,state] == 1:   rowStr += 'X'                 # simplified check if it is terminal/pit state
            elif a == 0:                rowStr += '<'
            elif a == 1:                rowStr += '>'
            elif a == 2:                rowStr += '^'
            else:                       rowStr += 'v'
            rowStr += ' '
        rowStr += '|'
        print(rowStr)

def PrintValues(world, Val):
    for row in reversed(range(world.Height)):
        rowStr = '| '
        for col in range(world.Width):
            state = world.Width * row + col
            if Val[state] >= 0:
                rowStr += ' {:1.3f} '.format(round(Val[state],3))
            else:
                rowStr += '{:1.3f} '.format(round(Val[state],3))
        rowStr += '|'
        print(rowStr)

def PrintQResults(world, Q_Values):
    policy = np.zeros(world.Size)
    Val = np.zeros(world.Size)
    for p in range(len(policy)):
        policy[p], Val[p] = BestActMaxQ(p, Q_Values)
    PrintPolicyVal(world, Val , policy)

def ShowVal(world, tail, title):
    plt.figure()
    plt.plot(tail)
    legend = []
    for row in range(world.Height):
        for col in range(world.Width):
            legend.append("[{},{}]".format(col+1,row+1))
    plt.legend(legend)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('{}.png'.format(title))
    plt.show()

def BestActMaxQ(state, Q_Values):
    bestAct = None
    maxQ = float('-inf')
    for act in range(len(Q_Values[0])):
        q = Q_Values[state,act]
        if q > maxQ:
            bestAct = act
            maxQ = q
    return bestAct, maxQ
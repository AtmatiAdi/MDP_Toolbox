import numpy as np
import world as World
MIN_DELTA = 1e-3

def maxV_BestAct(world, state, V):
    bestAct = None
    maxV = float('-inf')
    for act in range(len(world.Actions)):                           # For every action
        sumV = 0
        for transState in range(world.Size):                        # For every transition
            sumV += world.P[act,state,transState] * V[transState]
        if sumV > maxV:
            maxV = sumV
            bestAct = act
    val = world.R[state] + world.Gamma * maxV
    return val, bestAct

def ValueIteration(world, iter):
    tail = []
    V = np.zeros(world.Size)                            # Initalize Values array
    A = np.zeros(world.Size)                            # Initalize Actions array
    for i in range(iter):                               # Iteration counting
        isChanging = False                              # For useless iteration breaker
        for state in  range(world.Size):                # For every state 
            if world.P[0,state,state] == 1:                   # Simplified check if it is terminal/pit state
                V[state] = world.R[state,]                   # Always starting from terminal states gets its final value
                continue
            newV, A[state] = maxV_BestAct(world, state, V) # get new values and Actions
            if np.abs(V[state] - newV) > MIN_DELTA:             # Checking if updates does something
                isChanging = True                               
            V[state] = newV                                     # Updating value
        if isChanging == False:                                 # If nothing changes break the loop
            tail.append(V.copy())
            break
        tail.append(V.copy())
    return V, A, tail
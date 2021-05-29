import numpy as np
import world as World

class Scene:
    World = 0
    Counter = 0

    def __init__(self, world):
        self.World = world
        self.Counter = np.zeros(shape=(world.Size,len(world.Actions)))
    def Move(self, state, action):
        newState = np.random.choice(range(self.World.Size),p=self.World.P[action,state])
        self.Counter[newState,action] += 1
        return newState
    def GetSize(self):              return self.World.Size
    def GetActions(self):           return self.World.Actions
    def Reset(self):                return self.World.Start
    def GetReward(self, state):     return self.World.R[state]
    def IsTerminate(self, state):   return True if self.World.P[0,state,state] == 1 else False   
    def GetCount(self, state, act): return self.Counter[state, act] if self.Counter[state, act] > 0 else 1

class Agent:
    Scene = 0
    Q_Values = 0
    State = 0
    Explor = 0
    Gamma = 0
    def __init__(self, scene, explor, gamma):
        self.Scene = scene
        # Initalize Q_Values array for every state and action
        self.Q_Values = np.zeros(shape=(scene.GetSize(),len(scene.GetActions())))
        self.State = scene.Reset()
        self.Explor = explor            # This is agents propertities
        self.Gamma = gamma              # This is agents propertities

    def ChooseAction(self):
        bestAct = None
        maxQ = float('-inf')
        if self.Explor > 0 and np.random.uniform(0,1) <= self.Explor:
            bestAct = np.random.choice(range(len(self.Q_Values[0])))
        else:
            for act in range(len(self.Q_Values[0])):
                q = self.Q_Values[self.State,act]
                if q > maxQ:
                    bestAct = act
                    maxQ = q
        return bestAct

    def Learn(self, iter):
        while iter > 0:
            if self.Scene.IsTerminate(self.State):                                 
                self.Q_Values[self.State] = self.Scene.GetReward(self.State)   # Assign terminal value to Q value of terminal state
                self.State = self.Scene.Reset()
                iter -= 1
            else:                                                           # We are still playing
                action = self.ChooseAction()                  # Choice best action asing on Q values
                newState = self.Scene.Move(self.State, action)                        # Perform action on the scene     
                self.Q_Values[self.State, action] = self.CalcQ(newState, action)                                          # Update Q walue of current state
                self.State = newState                                                        # Realize where I am in my mind    
        return self.Q_Values

    def CalcQ(self, newState, action):
        oldQ = self.Q_Values[self.State, action]    
        tranQ = max(self.Q_Values[newState]) 
        reward = self.Scene.GetReward(self.State)
        learn = 1 / self.Scene.GetCount(self.State, action)
        return oldQ + learn * (reward + self.Gamma * tranQ - oldQ)   


def QLearning(world, iter):
    scene = Scene(world)
    agent = Agent(scene, world.Explor, world.Gamma)

    return agent.Learn(iter) 
import numpy as np
import GridWorldMDP as MDP
import matplotlib.pyplot as plt
#TODO
class Solver:


    def __init__(self, epsilon, num_episodes, gamma, xdim, ydim):
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.xdim = xdim
        self.ydim = ydim
        self.Qdict = {}


    # to define current state as (s[0],s[1])
    def epsilon_greedy(self, state):
        random_seed = np.random.random()

        if random_seed < self.epsilon:
            action = np.random.choice(MDP.actions(state))
        else:
            state_Qdict = {action: self.Q(state,action) for action in MDP.actions(state)}
            action = max(state_Qdict, key=state_Qdict.get)


        return action

    def init_Qdict(self):
        state_actions = []
        for x in range(0,self.xdim):
            for y in range(0,self.ydim):
                for a in ['Up', 'Down', 'Left','Right']:
                    state_actions.append( ((x,y), a) )


        for k in state_actions:
            self.Qdict[k] = 0

        print(self.Qdict)


    def Q(self, state,action):
        # return the Q value
        return self.Qdict[(tuple(state),action)]



    def Qlearning(self):
        # initialize Q function
        self.init_Qdict()
        reward_trj = []
        for i in range(0, self.num_episodes):
            # print(f"=========episode {i}========")
            reward_Episode = 0

            # random start state
            startState = [np.random.randint(0, xdim-1), np.random.randint(0, ydim-1)]
            currState = startState.copy()
            MDP_state = startState.copy()
            time = 0
            MDP_state.append(time)
            # print("MDP state:", MDP_state)
            # print("current state:", currState)


            while not MDP.IsEnd(MDP_state):
                print(f"current MDP_state: {MDP_state}")

                #eplison greedy actions
                action = self.epsilon_greedy(currState)
                # successor state
                succState_reward = MDP.succAndReward(MDP_state, action)
                succState = succState_reward[:2]
                reward = float(succState_reward[-1])


                succState_Qdict = [self.Q(succState, action) for action in MDP.actions(succState)]
                self.Qdict[(tuple(currState),action)] = self.Qdict[(tuple(currState),action)] + self.gamma * (reward + max(succState_Qdict) - self.Qdict[(tuple(currState),action)])

                currState = succState.copy()


                MDP_state = succState_reward.copy()[:3]
                reward_Episode = reward_Episode + reward

            # sum_reward_Ep = sum_reward_Ep + reward_Episode
            # reward_trj.append(sum_reward_Ep/float(i+1))
            reward_trj.append(reward_Episode)

        return reward_trj




if __name__ == '__main__':
    epsilon = 0.1
    num_episodes = 500
    gamma = 1
    xdim = 12
    ydim = 4
    gw = Solver(epsilon, num_episodes, gamma, xdim, ydim)


    reward_arr=np.zeros(num_episodes)
    for r in range(0,10):
        reward_trj = gw.Qlearning()
        reward_arr = reward_arr + np.array(reward_trj)
    reward_arr = reward_arr/float(10)
    x = [i for i in range(0,num_episodes)]
    plt.plot(x,reward_arr)
    plt.show()
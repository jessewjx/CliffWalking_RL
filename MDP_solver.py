import numpy as np
import GridWorldMDP as MDP
import matplotlib.pyplot as plt


class Solver:
    def __init__(self, epsilon, num_episodes, alpha, gamma, xdim, ydim):
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.xdim = xdim
        self.ydim = ydim
        self.Qdict = {}

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

    def Q(self, state,action):
        # return the Q value
        return self.Qdict[(tuple(state),action)]

    def Qlearning(self):
        # initialize Q function
        self.init_Qdict()
        cumReward_trajectory = []

        # roll out trajectories
        for i in range(0, self.num_episodes):
            reward_Episode = 0

            # random start state
            # state in Q function is the position [x,y]
            # MDP state is [x,y,t]
            startState = [np.random.randint(0, xdim-1), np.random.randint(0, ydim-1)]
            currState = startState.copy()
            MDP_state = startState.copy()
            time = 0
            MDP_state.append(time)

            # roll out an episode until reaching the end state
            # the checking the MDP state that involves the time step
            while not MDP.IsEnd(MDP_state):

                #each step take eplison greedy action
                action = self.epsilon_greedy(currState)

                # successor state, reward in the MDP model
                # succState_reward = [x,y,t,reward]
                succState_reward = MDP.succAndReward(MDP_state, action)
                succState = succState_reward[:2]
                reward = float(succState_reward[-1])

                # update value of Q function of current state,action
                # with the reward and the value of successor state
                succState_Qdict = [self.Q(succState, action) for action in MDP.actions(succState)]
                self.Qdict[(tuple(currState),action)] = self.Qdict[(tuple(currState),action)] + self.alpha * (reward + self.gamma * max(succState_Qdict) - self.Qdict[(tuple(currState),action)])

                # state transition
                currState = succState.copy()
                MDP_state = succState_reward.copy()[:3]

                # collect reward along the trajectory
                reward_Episode = reward_Episode + reward


            cumReward_trajectory.append(reward_Episode)

        return cumReward_trajectory

    def Sarsa(self):
        # initialize Q function
        self.init_Qdict()
        cumReward_trajectory = []

        # roll out trajectories
        for i in range(0, self.num_episodes):
            reward_Episode = 0

            # random start state
            # state in Q function is the position [x,y]
            # MDP state is [x,y,t]
            startState = [np.random.randint(0, xdim - 1), np.random.randint(0, ydim - 1)]
            currState = startState.copy()
            MDP_state = startState.copy()
            time = 0
            MDP_state.append(time)

            # roll out an episode until reaching the end state
            # the checking the MDP state that involves the time step
            while not MDP.IsEnd(MDP_state):
                # each step take eplison greedy action
                action = self.epsilon_greedy(currState)

                # successor state, reward in the MDP model
                # succState_reward = [x,y,t,reward]
                succState_reward = MDP.succAndReward(MDP_state, action)
                succState = succState_reward[:2]
                reward = float(succState_reward[-1])

                # update value of Q function of current state,action
                # with the reward and the Q value of successor state and epsilon greedy action
                succAction = self.epsilon_greedy(succState)
                self.Qdict[(tuple(currState), action)] = self.Qdict[(tuple(currState), action)] + self.alpha * (
                            reward + self.gamma * self.Qdict[(tuple(succState), succAction)] - self.Qdict[
                        (tuple(currState), action)])

                # state transition
                currState = succState.copy()
                MDP_state = succState_reward.copy()[:3]

                # collect reward along the trajectory
                reward_Episode = reward_Episode + reward

            cumReward_trajectory.append(reward_Episode)

        return cumReward_trajectory

    def Visualize_learned_policy(self):

        for x in range(0, self.xdim):
            for y in range(0, self.ydim):
                state = [x,y]
                state_Qdict = {action: self.Q(state, action) for action in MDP.actions(state)}
                action = max(state_Qdict, key=state_Qdict.get)





if __name__ == '__main__':
    epsilon = 0.1
    num_episodes = 50
    gamma = 1
    alpha = 1
    xdim = 12
    ydim = 4
    gw = Solver(epsilon, num_episodes, alpha, gamma, xdim, ydim)

    # Q learning
    Qreward_arr=np.zeros(num_episodes)
    # average the episodes rewards over runs
    for r in range(0,10):
        Qreward_trj = gw.Qlearning()
        Qreward_arr = Qreward_arr + np.array(Qreward_trj)
    Qreward_arr = Qreward_arr/float(10)


    # Sarsa learning
    Sreward_arr=np.zeros(num_episodes)
    for r in range(0,10):
        Sreward_trj = gw.Sarsa()
        Sreward_arr = Sreward_arr + np.array(Sreward_trj)
    Sreward_arr = Sreward_arr/float(10)


    # plot average reward as a function of episodes
    x = [i for i in range(0, num_episodes)]
    plt.plot(x,Qreward_arr,label = "Q learning")
    plt.plot(x, Sreward_arr,label = "Sarsa")
    plt.xlabel('episode count')
    plt.ylabel('cummulative reward')
    plt.legend()
    plt.show()
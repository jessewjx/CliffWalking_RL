import numpy as np
import GridWorldMDP as MDP
import matplotlib.pyplot as plt


class Solver:
    def __init__(self, epsilon, num_episodes, alpha, gamma, xdim, ydim, varying_epsilon, initial_Q):
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.xdim = xdim
        self.ydim = ydim
        self.varying_epsilon = varying_epsilon
        self.initial_Q = initial_Q
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
            self.Qdict[k] = self.initial_Q


    def Q(self, state,action):
        # return the Q value
        return self.Qdict[(tuple(state),action)]


    def Qlearning(self):
        # initialize Q function
        self.init_Qdict()
        cumReward_trajectory = []

        # roll out trajectories
        for i in range(0, self.num_episodes):
            # varying epsilon or not
            if self.varying_epsilon and i > 0:
                self.epsilon = self.epsilon - 0.1/500


            reward_Episode = 0

            # start from [0,0]
            # state in Q function is the position [x,y]
            # MDP state is [x,y,t]
            startState = [0,0]
            currState = startState.copy()
            MDP_state = startState.copy()
            time = 0
            MDP_state.append(time)

            # roll out an episode until reaching the end state
            # the checking the MDP state that involves the time step
            while not MDP.IsEnd(MDP_state):
                time = time + 1
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
            # varying epsilon or not
            if self.varying_epsilon and i > 0:
                self.epsilon = self.epsilon - 0.1 / 500

            reward_Episode = 0

            # random start state
            # state in Q function is the position [x,y]
            # MDP state is [x,y,t]
            # startState = [np.random.randint(0, xdim - 1), np.random.randint(0, ydim - 1)]
            startState = [0,0]
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
        for y in reversed(range(self.ydim)):
            x_actions = []
            for x in range(0,self.xdim):
                state = [x,y]
                state_Qdict = {action: self.Q(state, action) for action in MDP.actions(state)}
                action = max(state_Qdict, key=state_Qdict.get)
                x_actions.append(action)
            print('|'.join("{:<8}".format(a) for a in x_actions))






if __name__ == '__main__':
    varying_epsion = False
    initial_Q = 0
    epsilon = 0.1
    num_episodes = 500
    num_runs = 10
    gamma = 1
    alpha = 0.5
    xdim = 12
    ydim = 4
    gw = Solver(epsilon, num_episodes, alpha, gamma, xdim, ydim,varying_epsion,initial_Q)

    if varying_epsion:
        print("Varying Epsilon...")
    else:
        print("Fixed Epsilon...")

    # Q learning
    print("start Q learning...")
    Qreward_arr=np.zeros(num_episodes)
    # average the episodes rewards over runs
    for r in range(0,num_runs):
        Qreward_trj = gw.Qlearning()
        Qreward_arr = Qreward_arr + np.array(Qreward_trj)
    Qreward_arr = Qreward_arr/float(num_runs)
    print("Visualization of Qlearning policy")
    gw.Visualize_learned_policy()

    # Sarsa learning
    print("\n\nstart Sarsa learning...")
    Sreward_arr=np.zeros(num_episodes)
    for r in range(0,num_runs):
        Sreward_trj = gw.Sarsa()
        Sreward_arr = Sreward_arr + np.array(Sreward_trj)
    Sreward_arr = Sreward_arr/float(num_runs)
    print("Visualization of Sarsa policy")
    gw.Visualize_learned_policy()


    # plot average reward as a function of episodes
    x = [i for i in range(0, num_episodes)]
    plt.plot(x,Qreward_arr,label = "Q learning")
    plt.plot(x, Sreward_arr,label = "Sarsa")
    plt.xlabel('episode count')
    plt.ylabel('cummulative reward')
    plt.ylim([-200, 0])
    plt.legend()
    plt.show()
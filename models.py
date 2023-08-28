import numpy as np

class State_Space_Model:
    """
    state space model parameterized by transition probabilities and an initial state.
    """

    def __init__(self, transitions = None, initial_states = None):

        """
        initialization
            @param transitions:     transition matrix p[a, s, s'] (binary action)
            @param initial_states:  initial state one hot vector
        """

        if transitions is None:
            raise error("empty transitions matrices")
            # implement something robust here
        self.transitions = transitions

        self.n_actions, self.n_states = self.transitions.shape[:-1]

        if initial_states is None:
            self.initial_states = np.zeros(self.n_states)
            indx = np.random.choice(self.n_states, size = 1)
            self.initial_states[indx] = 1
        else:
            self.initial_states = initial_states

        self.current_states = self.initial_states

    def step(self, a):
        '''
        stochastically simulate the state space model

            @param a: action 0/1
        '''
        current_states = self.current_states.reshape((1,-1))
        pos_state = (current_states @ self.transitions[a]).reshape(-1)

        # sample through p[s'|a]
        current_states = np.zeros(self.n_states)
        indx =  np.random.choice(self.n_states, size=1, p = pos_state)

        current_states[indx] = 1

        self.current_states = current_states

    def set_transitions(self, new_transitions):
        self.transitions = new_transitions

    def get_transitions(self):
        return self.transitions

    def get_current_states(self):
        return self.current_states

    def reset(self):
        self.current_states = self.initial_states


class Restless_Bandit(State_Space_Model):

    """
    N-arm restless bandit, each arm is modeled as a state space model
    """

    def __init__(self, transitions = None, initial_states = None, R = None):
        """
        initialization

        @param transitions:     transition matrices [n_arms, n_actions, n_state, n_state]
        @param initial_states:  [n_arms, n_state] (one-hot)
        @param R:               rewards [n_state, ]

        """

        if transitions is None:
            raise error("not implemented")
            # implement something robust here
        self.transitions = transitions

        self.n_arms, self.n_actions, self.n_states = transitions.shape[:-1]
        self.initial_states = initial_states

        self.arms = list()
        for transition, initial in zip(self.transitions, self.initial_states):
            self.arms.append(State_Space_Model(transitions = transition, initial_states = initial))

        # house keeping
        self.current_states = initial_states
        self.current_reward = 0
        self.R = R


    def step(self, actions):
        """
        stochastically simulate each arm for one step.

            @param actions: one hot vector [n_arms, ] (can be multiple ones)

        """

        self.current_states = np.zeros((self.n_arms, self.n_states))

        for i, (arm, a) in enumerate(zip(self.arms, actions)):
            arm.step(a)
            self.current_states[i,:] = arm.get_current_states()

        current_state_indx = self.current_states.argmax(axis = 1)
        self.current_reward = (self.R[current_state_indx] * actions).sum()


    def reset(self):
        self.current_reward = 0
        for arm in self.arms:
            arm.reset()
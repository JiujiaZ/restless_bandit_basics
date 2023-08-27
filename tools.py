import numpy as np

def random_transitions(n_actions = 2, n_states = 2):

    """
        @return transitions: transition probability p[a,s,s']
    """

    transitions = np.zeros((n_actions, n_states, n_states))

    for a in range(n_actions):
        temp = np.random.uniform(low=0, high=1, size=(n_states,n_states))
        temp = temp / temp.sum(axis = 1, keepdims = True)
        transitions[a,:,:] = temp
    return transitions




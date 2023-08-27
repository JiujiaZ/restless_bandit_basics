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


def noisy_transitions(transition, sigma):

    """
    add noise to provided transition matrices
    
        @param transition: transition matrix p[a, s, s']
        @param sigma:      noise level N(0, sigma)
        
        @return:
            transition: perturbed transition matrix hat{p}[a, s, s']
    """

    n_actions, n_states = transition.shape[:-1]

    for a in range(n_actions):
        epsilon = np.random.normal(loc=0.0, scale=sigma, size=n_states**2)
        epsilon = epsilon.reshape((n_states, n_states))

        transition[a] += epsilon
        transition[a] /= transition.sum(axis = 1, keepdims=True)

    return transition





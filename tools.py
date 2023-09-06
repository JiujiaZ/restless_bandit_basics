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


def make_valid_transition(transitions):
    """
    modify any input transition to  a 'valid' transition
    assuming 2 state.

    @param transitions: P[a, s, s'], sum_s' = 1

    s: 0 bad state, 1 good state

        1. acting is good
        2. starting good state more likely to stay good
   """

    n_actions, n_states = transitions.shape[:-1]

    assert n_actions == 2
    assert n_states == 2

    # check acting is good, otherwise swap:
    # (a = 1, s, s' = 1) > (a = 1, s, s' = 0)
    for s in range(n_states):
        if transitions[1, s, 1] <= transitions[1, s, 0]:
            transitions[1, s, 0] = transitions[1, s, 1] * np.random.rand()
            transitions[1, s, 1] = 1 - transitions[1, s, 0]

    # check starting good is more likely to stay at good state:
    # (a, s = 1, s' = 1) > (a, s = 0, s' = 1)
    for a in range(n_actions):
        if transitions[a, 1, 1] <= transitions[a, 0, 1]:
            transitions[a, 0, 1] = transitions[a, 1, 1] * np.random.rand()
            # make sure sums up to 1
            transitions[a, 0, 0] = 1 - transitions[a, 0, 1]

    assert np.allclose(transitions.sum(axis = -1), 1)

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
    perturbed_transition = np.zeros_like(transition)

    for a in range(n_actions):
        epsilon = np.random.normal(loc=0.0, scale=sigma, size=n_states**2)
        epsilon = epsilon.reshape((n_states, n_states))

        perturbed_transition[a] = abs(transition[a] + epsilon) # prevent negative
        perturbed_transition[a] /= perturbed_transition[a].sum(axis = 1, keepdims=True)

    return perturbed_transition

def random_features(n_arms, n_dims, scale = 1, sigma = 1e-1):
    """
    generate random features per arm for linear contextual bandit

    @param n_arms: number of arms
    @param n_dims: feature dimension per arm
    @param scale:  maximum L2 norm bound
    @param sigma: standard deviation

    @return feature

    """

    feature = np.random.normal(scale = sigma, size = (n_arms, n_dims) )
    if np.linalg.norm(feature) > scale :
        feature = feature / ( np.linalg.norm(feature) * scale )

    return feature


def logsumexp_trick(x):
    """
    For exp(x_i) / sum_i exp(x_i) avioding over flow
    """

    c = x.max()
    y = c + np.log( np.sum(np.exp(x-c) ) )

    return np.exp(x - y)




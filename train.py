import numpy as np
from policies import whittle_index

def WI_train(rb, R, transitions = None, episode=100, K=1):
    """
    train with a whittle indexing, assume transition dynamics stays the same

        @param rb:          an instance of restless_bandit
        @param R:           reward for each states
        @param transitions: transition matrix p[a, s, s'], if provided WI will be computed based on
                            this value, otherwise using rb.transitions
        @param episode:     episode number

        @return rewards:    a list of rewards
    """

    n_arms, n_actions, n_states = rb.transitions.shape[:-1]
    rewards = list()

    # compute all Whittle index based on explictly provided transitions or rb parameters
    WI = np.zeros((n_arms, n_states))
    if transitions is None:
        transitions = rb.transitions
    for n in range(n_arms):
        for s in range(n_states):
            WI[n, s] = whittle_index(transitions[n], s, R = R, gamma=0.99, lb=-1, ub=1, subsidy_break=-1)

    for e in range(episode):
        # rank preferred arm from high to low
        indx = WI[rb.current_states.astype('bool')]
        indx = indx.argsort()[::-1][:K]

        actions = np.zeros(n_arms, dtype=int)
        actions[indx] = 1

        rb.step(actions=actions)
        rewards.append(rb.current_reward)

    return rewards


def random_train(rb, episode=100, K=1):
    """
    train with random selection

        @param rb:          an instance of restless_bandit
        @param episode:     episode number

        @return rewards:    a list of rewards
    """

    n_arms, n_actions, n_states = rb.transitions.shape[:-1]
    rewards = list()

    for e in range(episode):
        # rank preferred arm from high to low
        indx = np.random.choice(n_arms, size=K, replace=False)

        actions = np.zeros(n_arms, dtype=int)
        actions[indx] = 1

        rb.step(actions=actions)
        rewards.append(rb.current_reward)

    return rewards


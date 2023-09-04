import numpy as np
from policies import whittle_index, uc_whittle
import copy

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

def UCW_train(rb, R, H = 20, episode=100, K=1):
    """
    train with a whittle indexing, assume transition dynamics stays the same

        @param rb:          an instance of restless_bandit
        @param R:           reward for each states
        @param H:           episode length
        @param episode:     episode number

        @return rewards:    a list of rewards
    """

    n_arms, n_actions, n_states = rb.transitions.shape[:-1]
    rewards = list()

    # initialize counts [n, a, s, s']
    counts = np.zeros((n_arms, n_actions, n_states, n_states))
    subsidy = 0

    for e in range(episode):
        WI = np.zeros((n_arms, n_states))
        rb.reset()
        for n in range(n_arms):

            s0 = rb.initial_states[n].argmax()

            est_P = uc_whittle(s0, subsidy, R = R, gamma = 0.99, counts = counts, n_arms = n_arms, t = e+1, delta = 1e-3)

            for s in range(n_states):
                WI[n, s] = whittle_index(est_P, s, R=R, gamma=0.99, lb=-10, ub=10, subsidy_break=-1)

        # rank preferred arm from high to low
        for h in range(H):

            current_states = copy.copy(rb.current_states)

            # arms being pulled
            indx = WI[current_states.astype('bool')]
            indx = indx.argsort()[::-1][:K]

            actions = np.zeros(n_arms, dtype=int)
            actions[indx] = 1

            rb.step(actions=actions)
            rewards.append(rb.current_reward)

            next_states = rb.current_states

            # update:
            for n in indx:
                counts[n, current_states[n].argmax(), next_states[n].argmax()] += 1

            subsidy = WI[current_states[indx[-1]]]

    return rewards
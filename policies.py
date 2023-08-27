import numpy as np
def value_iteration(transitions, R, lamb_val, gamma, epsilon=1e-2):

    """
    value iteration for a MDS:

        @param transitions: transition matrix p[a, s, s']
        @param R:           reward for all states R[s]
        @param lamb_val:    lagrangian multiplier associated with formulation of WI
        @param gamma:       discounted factor for reward
        @param epsilon:     tolerance to terminate value iteration

        @return Q_func:     Q_values for each state and actions [s, a]

    """

    n_actions, n_states = transitions.shape[:-1]
    value_func = np.random.rand(n_states)
    delta = np.ones((n_states))
    iters = 0

    while np.max(delta) >= epsilon:
        iters += 1
        orig_value_func = np.copy(value_func)

        Q_func = np.zeros((n_states, n_actions))
        for s in range(n_states):

            for a in range(n_actions):
                Q_func[s, a] += - a * lamb_val + R[s] + gamma * np.dot(transitions[a, s, :], value_func)
            value_func[s] = np.max(Q_func[s, :])

        delta = np.abs(orig_value_func - value_func)

    return Q_func


def whittle_index(transitions, state, R, gamma, lb, ub, subsidy_break, epsilon=1e-4):
    """
    whittle index for a specified state using binary search: https://arxiv.org/pdf/2205.15372.pdf

        @param transitions:     transition matrix p[a, s, s']
        @param state:           a single specified state \in [S]
        @param R:               reward for all states R[s]
        @param lamb_val:        lgrangian multiplier associated with formulation of WI
        @param gamma:           discounted factor for reward
        @param lb, ub:          initial lower / upper bound of WI
        @param subsidy_break:   lower tolerance of WI (if returned, decrease lb)
        @param epsilon:         tolerance to terminate binary search

        @return subsidy:        approximated whittle index for specified state
    """

    while abs(ub - lb) > epsilon:
        lamb_val = (lb + ub) / 2
        # print('lamb', lamb_val, lb, ub)

        # need to adjust (lower) initial lb
        if ub < subsidy_break:
            print('breaking early!', subsidy_break, lb, ub)
            return -np.inf

        Q_func = value_iteration(transitions, R, lamb_val, gamma)

        # binary search:
        action = np.argmax(Q_func[state, :])

        # Q(s, 0) > Q(s, 1)_{lamb_val}: lamb_val in smaller interval
        if action == 0:
            ub = lamb_val
        # Q(s, 0) < Q(s, 1)_{lamb_val}: lamb_val in bigger interval
        elif action == 1:
            lb = lamb_val
        else:
            raise Error(f'action not binary: {action}')

    subsidy = (ub + lb) / 2
    return subsidy
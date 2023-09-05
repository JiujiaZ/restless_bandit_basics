import numpy as np
import gurobipy as gb
from gurobipy import GRB


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

def est_confidence(counts, n_arms, t =1, delta = 1e-3):
    """
    confindence ball at time t for a single arm

    @param counts: N[a, s, s']
    @param delta: positive

    @return diam:    diam[s,a]
    """
    n_actions, n_states = counts.shape[:-1]

    # emperical transitions p[a, s, s']
    N_a_s = counts.sum(axis = -1)

    never_observed = np.where(N_a_s == 0)
    observed = np.where(N_a_s > 0)

    est_transition = np.zeros_like(counts)

    for a, s in zip(*never_observed):
        est_transition[a, s, :] = 1 / n_states
    for a, s in zip(*observed):
        est_transition[a, s, :] = counts[a, s, :] / N_a_s[a, s]

    # confidentce radius d[s, a]
    diam = ((2 * n_states * np.log(2 * n_states * n_actions * n_arms * t**4 / delta ) ) / np.maximum(1, N_a_s )) ** 0.5

    return est_transition, diam

def uc_whittle(s0, subsidy, R, gamma, counts, n_arms, t =1, delta = 1e-3):


    est_p, diam = est_confidence(counts, n_arms, t = t, delta = delta)

    n_actions, n_states = est_p.shape[:-1]

    model = gb.Model('UCWhittle')

    # variable for value functions
    value_sa = [[model.addVar(vtype=GRB.CONTINUOUS, name=f'v_{s}_{a}')
                 for a in range(n_actions)] for s in range(n_states)]
    value_s = [model.addVar(vtype=GRB.CONTINUOUS, name=f'v_{s}')
               for s in range(n_states)]

    # dummy variables for confidence interval d_a_s_s':
    d = [[[model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=diam[s, a],
                        name=f'd_{a}_{s}_{s_prime}'
                        ) for s_prime in range(n_states)]
          for s in range(n_states)]
         for a in range(n_actions)]

    # p_a_s_s':
    p = [[[model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                        name=f'p_{a}_{s}_{s_prime}'
                        ) for s_prime in range(n_states)]
          for s in range(n_states)]
         for a in range(n_actions)]

    # define constraints -------------------------------------------------------
    for a in range(n_actions):
        for s in range(n_states):

            for s_prime in range(n_states):
                model.addConstr(p[a][s][s_prime] <= est_p[a, s, s_prime] + d[a][s][s_prime])
                model.addConstr(p[a][s][s_prime] >= est_p[a, s, s_prime] - d[a][s][s_prime])
                model.addConstr(d[a][s][s_prime] <= est_p[a, s, s_prime])
                model.addConstr(d[a][s][s_prime] <= 1 - est_p[a, s, s_prime])

            # 1 norm constrant
            model.addConstr(sum(d[a][s]) <= diam[s, a])
            # probability sums up to 1
            model.addConstr(sum(p[a][s]) == 1)
            # q function q(s,a)
            model.addConstr(value_sa[s][a] == -subsidy * a + R[s]
                            + gamma * sum(list(map(lambda x, y: x * y, value_s, p[a][s])))
                            )

            # value function v(s):
            if a == 0:
                model.addConstr(value_s[s] == gb.max_(value_sa[s]))

    # set objective:
    model.setObjective(value_s[s0], GRB.MAXIMIZE)

    # grb model parameters:
    model.write('UCWhittle.lp')
    model.setParam('NonConvex', 2)  # nonconvex constraints
    # model.setParam('DualReductions', 1)
    max_iterations = 10000
    model.setParam('IterationLimit', max_iterations)

    # optimize
    model.optimize()

    # get est_p
    new_p = np.zeros_like(est_p)

    for a in range(n_actions):
        for s in range(n_states):
            for s_prime in range(n_states):
                new_p[a, s, s_prime] = p[a][s][s_prime].X

    if model.status != GRB.OPTIMAL:
        print('not optimal solution')

    return new_p






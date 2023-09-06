import numpy as np
from policies import whittle_index, uc_whittle
from tools import *
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
        rb.reset()

        for h in range(H):
            WI = np.zeros((n_arms, n_states))
            for n in range(n_arms):

                s0 = rb.initial_states[n].argmax()

                est_P = uc_whittle(s0, subsidy = subsidy, R = R, gamma = 0.99, counts = counts[n], n_arms = n_arms, t = e+1, delta = 1e-3)

                for s in range(n_states):
                    WI[n, s] = whittle_index(est_P, s , R=R, gamma=0.99, lb=-10, ub=10, subsidy_break=-1)


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
            for n in indx: # arms, where a = 1
                counts[n, 1, current_states[n].argmax(), next_states[n].argmax()] += 1
            for n in set(np.arange(n_arms)) - set(indx): # arms, where a = 0
                counts[n, 0, current_states[n].argmax(), next_states[n].argmax()] += 1

            smallest_arm = indx[-1]
            smallest_state =  current_states[indx[-1]].argmax()
            subsidy = WI[smallest_arm,smallest_state]

            print(f' h = {h}, subsidy = {subsidy}')

    return rewards


def Exp3_train(rb, R, episode=100, K=1, contextual = False, n_dims = 10):
    """
    EXP3 algorithm for bandit (maximize reward)

        @param rb:          an instance of restless_bandit
        @param R:           reward for each states [0-1] per state
        @param episode:     episode number

        @return rewards:    a list of rewards
    """

    n_arms, n_actions, n_states = rb.transitions.shape[:-1]
    rewards = list()

    p = np.ones( n_arms ) / n_arms
    cum_reward = np.zeros(n_arms)

    # optimal learning rate:
    eta = np.sqrt( 2 * np.log( n_arms ) / (episode * n_arms) )

    if contextual: theta_star = gaussian_features(n_dims, 1)

    for e in range(episode):

        # sample arms
        indx = np.random.choice(n_arms, size=K, replace=False, p = p)

        actions = np.zeros(n_arms, dtype=int)
        actions[indx] = 1

        rb.step(actions=actions)

        if contextual:
            # get random feature
            X = random_features(n_arms, n_dims - 1, scale=1 / np.sqrt(2))
            # attach states
            if e == 0:
                prev_observed_states = rb.current_states.argmax(axis=-1, keepdims=True)
            else:
                prev_observed_states = np.ones((n_arms, 1)) * 0.5
                prev_observed_states[indx] = rb.current_states.argmax(axis=-1, keepdims=True)[indx]
            X = np.hstack((prev_observed_states / 2 , X))

            # X = np.hstack((rb.current_states.argmax(axis=-1, keepdims=True) * 1 / 2, X))
            # X = X[:, :, np.newaxis]

            reward = (X[indx] @ theta_star).item() + np.random.normal(scale = 1e-1)
        else:
            reward = rb.current_reward

        rewards.append(reward)

        # adjusted reward
        R_hat = np.zeros(n_arms)
        observed_states = rb.current_states[indx].argmax(axis = -1)
        R_hat[indx] = reward / p[indx]
        # R_hat[indx] = R[observed_states] / p[indx]


        # exp3 update:
        cum_reward += R_hat
        # avoid over flow use log-sum-exp trick
        # p = np.exp(eta * cum_reward)
        # p = p/p.sum()
        p = logsumexp_trick(eta * cum_reward)

    return rewards


def UCB(rb, R, episode=100, K=1):
    """
    UCB algorithm for bandit (maximize reward) for a single arm

        @param rb:          an instance of restless_bandit
        @param R:           reward for each states [0-1] per state
        @param episode:     episode number

        @return rewards:    a list of rewards
    """

    assert K == 1, 'UCB only valid for pulling a single arm.'

    n_arms, n_actions, n_states = rb.transitions.shape[:-1]
    rewards = list()

    est_reward = np.zeros(n_arms)
    pull_count = np.zeros(n_arms)

    # alpha > 2
    alpha = 3


    for e in range(episode):

        # sample arms
        never_pulled = pull_count == 0
        if never_pulled.sum() >= 1: # some arms were never pulled
            indx = np.random.choice(n_arms, size = K, replace=False, p = never_pulled/ never_pulled.sum() )
        else:
            indx = (est_reward + np.sqrt(2 * alpha * np.log(e+1) / pull_count )).argmax()

        actions = np.zeros(n_arms, dtype=int)
        actions[indx] = 1

        rb.step(actions=actions)
        rewards.append(rb.current_reward)

        # update
        pull_count[indx] += 1
        est_reward[indx] = (est_reward[indx] * (pull_count[indx]-1) + rb.current_reward) / pull_count[indx]

    # print(pull_count)

    return rewards


def LinUCB_disjoint(rb, episode=100, K=1, n_dims = 10, common = True):
    """
    LinUCB (disjoint) algorithm for bandit (maximize reward) for a single arm
    Currently support random features

        @param rb:          an instance of restless_bandit
        @param episode:     episode number
        @param n_dims:      random feature dimension
        @param common:      (bool) true: optimal weights same for each arm

        @return rewards:    a list of rewards
    """

    assert K == 1, 'UCB only valid for pulling a single arm.'

    n_arms, n_actions, n_states = rb.transitions.shape[:-1]
    rewards = list()

    # alpha:
    alpha = 4

    # N[a, d, d]
    A = np.tile( np.eye(n_dims), (n_arms, 1, 1))
    b = np.zeros((n_arms, n_dims, 1))

    # needs to be implemented in model
    if common:
        theta_star = gaussian_features(1, n_dims)
        theta_star = np.tile(theta_star.squeeze(), (n_arms, 1) ) #[n, n_dims]
    else:
        theta_star = gaussian_features(n_arms, n_dims)

    theta_star = theta_star[:,:,np.newaxis]

    for e in range(episode):
        # get random feature
        X = random_features(n_arms, n_dims-1, scale = 1/np.sqrt(2))
        # attach states
        if e == 0:
            prev_observed_states = rb.current_states.argmax(axis=-1, keepdims=True)
        else:
            prev_observed_states = np.ones((n_arms, 1)) * 0.5
            prev_observed_states[indx] = rb.current_states.argmax(axis=-1, keepdims=True)[indx]
        X = np.hstack((prev_observed_states / 2 , X))

        # X = np.hstack((rb.current_states.argmax(axis = -1, keepdims = True) * 1/2, X))
        X = X[:, :, np.newaxis]

        # estimation
        theta = np.zeros((n_arms, n_dims, 1))
        p = np.zeros(n_arms)
        for a in range(n_arms):
            A_inv = np.linalg.pinv(A[a])
            theta[a] = A_inv @ b[a]
            p[a] = (theta[a].T @ X[a] + alpha * np.sqrt( X[a].T @ A_inv @ X[a] )).item()

        # select arms
        indx = p.argmax()
        actions = np.zeros(n_arms, dtype=int)
        actions[indx] = 1

        rb.step(actions=actions)
        # need to integrate this to model:
        current_reward = (X[indx].T @ theta_star[indx]).item() + np.random.normal(scale = 1e-1)
        rewards.append(current_reward)

        # update:
        A[indx] += X[indx] @ X[indx].T
        b[indx] += current_reward * X[indx]

    return rewards

def RoLinUCB(rb, episode=100, K=1, n_dims = 10, sigma_0 = 1e-1, sigma = 1e-1, lamb = 1e-1):
    """
    https://arxiv.org/pdf/2210.14483.pdf#page5

    @param rb:
    @param episode:
    @param K:
    @param n_dims:
    @param sigma_0:
    @param sigma:
    @param lamb:

    @return:
    """
    assert K == 1, 'UCB only valid for pulling a single arm.'

    n_arms, n_actions, n_states = rb.transitions.shape[:-1]
    rewards = list()






    return






def LEADER():
    """
    https://arxiv.org/pdf/2104.03781.pdf
    representation selection

    @return:
    """

    raise NotImplemented







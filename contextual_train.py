# use common state-independet features per iterate
import copy
import numpy as np
from tools import *

class contextual_random():

    def __init__(self, rb, theta_star, K = 1):
        self.rb = rb
        self.theta_star = theta_star
        self.K = K
        self.n_arms = rb.transitions.shape[0]

    def get_full_feature(self, partial_feature):
        prev_pulled = self.rb.pulled == 1  # [n_arms, ]
        prev_observed_states = np.ones((self.n_arms, 1)) * 0.5
        prev_observed_states[prev_pulled] = self.rb.current_states.argmax(axis=-1, keepdims=True)[prev_pulled]

        full_feature = np.hstack((prev_observed_states / 2, partial_feature))

        return full_feature

    def step(self, partial_feature):

        full_feature = self.get_full_feature(partial_feature)

        indx = np.random.choice(self.n_arms, size=self.K, replace=False)
        actions = np.zeros(self.n_arms, dtype=int)
        actions[indx] = 1

        self.rb.step(actions=actions)
        self.reward = (full_feature[indx] @ self.theta_star).item() + np.random.normal(scale=1e-1)



class contextual_exp3():

    def __init__(self, rb, theta_star, T, K = 1):
        self.rb = rb
        self.theta_star = theta_star
        self.K = K
        self.n_arms = rb.transitions.shape[0]
        self.p = np.ones( self.n_arms ) / self.n_arms
        self.cum_reward = np.zeros(self.n_arms)
        self.eta = np.sqrt(2 * np.log(self.n_arms) / (T * self.n_arms))

    def get_full_feature(self, partial_feature):
        prev_pulled = self.rb.pulled == 1  # [n_arms, ]
        prev_observed_states = np.ones((self.n_arms, 1)) * 0.5
        prev_observed_states[prev_pulled] = self.rb.current_states.argmax(axis=-1, keepdims=True)[prev_pulled]

        full_feature = np.hstack((prev_observed_states / 2, partial_feature))

        return full_feature

    def step(self, partial_feature):

        full_feature = self.get_full_feature(partial_feature)

        indx = np.random.choice(self.n_arms, size=self.K, replace=False, p=self.p)
        actions = np.zeros(self.n_arms, dtype=int)
        actions[indx] = 1

        self.rb.step(actions=actions)
        self.reward = (full_feature[indx] @ self.theta_star).item() + np.random.normal(scale=1e-1)

        # update:
        # adjusted reward
        R_hat = np.zeros(self.n_arms)
        observed_states = self.rb.current_states[indx].argmax(axis=-1)
        R_hat[indx] = self.reward / self.p[indx]

        # exp3 update:
        self.cum_reward += R_hat
        # avoid over flow use log-sum-exp trick
        self.p = logsumexp_trick(self.eta * self.cum_reward)



class contextual_LinUCB():

    def __init__(self, rb, theta_star, K = 1):
        self.rb = rb
        self.theta_star = theta_star
        self.K = K
        self.n_arms = rb.transitions.shape[0]
        # alpha:
        self.alpha = 4
        self.n_dims = theta_star.shape[0]

        # N[a, d, d]
        self.A = np.tile(np.eye(self.n_dims), (self.n_arms, 1, 1))
        self.b = np.zeros((self.n_arms, self.n_dims, 1))

    def get_full_feature(self, partial_feature):
        prev_pulled = self.rb.pulled == 1  # [n_arms, ]
        prev_observed_states = np.ones((self.n_arms, 1)) * 0.5
        prev_observed_states[prev_pulled] = self.rb.current_states.argmax(axis=-1, keepdims=True)[prev_pulled]

        full_feature = np.hstack((prev_observed_states / 2, partial_feature))

        return full_feature[:,:, np.newaxis]

    def step(self, partial_feature):

        X = self.get_full_feature(partial_feature)

        theta = np.zeros((self.n_arms, self.n_dims, 1))
        p = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.pinv(self.A[a])
            theta[a] = A_inv @ self.b[a]
            p[a] = (theta[a].T @ X[a] + self.alpha * np.sqrt(X[a].T @ A_inv @ X[a])).item()

        # select arms
        indx = p.argmax()
        actions = np.zeros(self.n_arms, dtype=int)
        actions[indx] = 1

        self.rb.step(actions=actions)

        print(X[indx].shape, self.theta_star.shape)
        self.reward = (X[indx].T @ self.theta_star).item() + np.random.normal(scale=1e-1)

        # update:
        self.A[indx] += X[indx] @ X[indx].T
        self.b[indx] += self.reward * X[indx]


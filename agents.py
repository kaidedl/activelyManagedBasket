import py_lets_be_quickly_rational
import finance
from state import State, States
from env import Env

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
from copy import deepcopy
from collections import deque


class PolicyRandom:
    def __init__(self, n_assets):
        self.n_actions = 1 + n_assets * (n_assets - 1)

    def action(self, states):
        n_sims = states.shape[0]
        return np.random.choice(self.n_actions, size=n_sims)


class PolicyNoChange:
    def action(self, states):
        n_sim = states.shape[0]
        return np.zeros(n_sim)


class PolicyDqn:

    def __init__(self, S, V, C, initial_weights, K, T):
        self.K = K
        self.T = T
        self.num_periods, self.num_sims, self.num_assets = S.shape
        self.num_actions = 1 + self.num_assets*(self.num_assets-1)
        self.num_flat_state = 3 * self.num_assets + self.num_assets * (self.num_assets-1) // 2 + 1

        self.epsilon = 0.1

        self.main_network = self.build_network()

        self.buffer = self.build_buffer(S, V, C, initial_weights, K, T)
        # target network?, replay buffer?

    def build_network(self):
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=self.num_flat_state))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model

    def epsilon_greedy(self, flat_state): # better boltzman
        q_values = self.main_network.predict(flat_state)
        greedy_actions = np.argmax(q_values, axis=1)
        random_actions = np.random.choice(self.num_actions, size=self.num_sims)
        eps = np.random.uniform(size=self.num_sims)
        return np.where(eps < self.epsilon, random_actions, greedy_actions)

    def build_buffer(self, S, V, C, initial_weights):
        flat_row, flat_col = np.triu_indices(self.num_assets, k=1)
        buffer = deque(maxlen=5000)
        weight = deepcopy(initial_weights)
        for i in range(self.num_periods-1):
            period = np.zeros((self.num_sims, self.num_periods))
            period[:, i] = 1
            flat_correlation = C[i][:, flat_row, flat_col].T
            X = np.concatenate([S[i], V[i], flat_correlation, weight, period], axis=1)
            action_indices = self.epsilon_greedy(X)
            updated_weight = Env.update_weight(S[i], weight, action_indices)
            next_period = np.zeros((self.num_sims, self.num_periods))
            next_period[:, i] = 1
            next_flat_correlation = C[i][:, flat_row, flat_col].T
            next_X = np.concatenate([S[i+1], V[i+1], next_flat_correlation, updated_weight, next_period], axis=1)
            done = i+1 == self.num_periods-1
            reward = np.maximum(0, np.sum(S[i+1]*updated_weight, axis=1) - self.K) if done else np.zeros(self.num_sims)
            for j in range(self.num_sims):
                buffer.append([X[j], action_indices[j], reward[j], next_X[j], done])
        return buffer

    def train(self):
        minibatch = self.buffer

        for state, action, reward, next_state, done in minibatch:
            if not done:
                target_Q = (reward + np.amax(self.target_network.predict(next_state)))
            else:
                target_Q = reward

            # compute the Q value using the main network
            Q_values = self.main_network.predict(state)

            Q_values[0][action] = target_Q

            # train the main network
            self.main_network.fit(state, Q_values, epochs=1, verbose=0)

    def action(self, states):
        pass


class PolicyGreedy:
    def __init__(self, K, heston_params, p_low, p_high, T, num_periods):
        self.K = K
        self.heston_params = heston_params
        self.p_low = p_low
        self.p_high = p_high
        self.T = T
        self.num_periods = num_periods

    def action(self, states):
        T_rem = self.T * (self.num_periods - states.time - 1) / self.num_periods
        actions, dist = q_approx_all(states, self.K, self.heston_params, T_rem, self.p_low, self.p_high)
        return np.argmax(dist, axis=1)


class PolicyLS:
    def __init__(self, policy, S, V, C, initial_weights, K, T):
        self.K = K
        self.T = T

        self.num_periods, self.num_sims, self.num_assets = S.shape
        self.num_actions = 1 + self.num_assets*(self.num_assets-1)
        self.states = []
        weights = deepcopy(initial_weights)
        for i in range(self.num_periods):
            self.states.append(States(i, S[i], V[i], C[i], weights))
            if i < self.num_periods-1:
                action_indices = policy.action(self.states[-1])
                weights = Env.update_weight(self.states[-1].spots, self.states[-1].weights, action_indices)

        self.flat_row, self.flat_col = np.triu_indices(self.num_assets, k=1)
        self.models = []
        self.train()

    def train(self):
        for i in range(self.num_periods-2, -1, -1):
            self.models.append([])
            flat_correlation = self.states[i].correlations[:, self.flat_row, self.flat_col].T
            T_rem = self.T * (self.num_periods - i - 1) / self.num_periods
            X = np.concatenate([self.states[i].spots, self.states[i].variances, flat_correlation,
                                self.states[i].weights], axis=1)
            F = PolynomialFeatures(2).fit_transform(X)
            for action in range(self.num_actions):
                action_indices = action * np.ones(self.num_sims)
                updated_weights = Env.update_weight(self.states[i].spots, self.states[i].weights, action_indices)
                if i==self.num_periods-2:
                    y = np.maximum(np.sum(updated_weights * self.states[i+1].spots, axis=1) - self.K, 0)
                else:
                    flat_correlation = self.states[i+1].correlations[:, self.flat_row, self.flat_col].T
                    next_X = np.concatenate([self.states[i+1].spots, self.states[i+1].variances, flat_correlation, updated_weights], axis=1)
                    next_F = PolynomialFeatures(2).fit_transform(next_X)
                    y = np.maximum(self.value(self.models[-2], next_F), axis=1)
                model = LinearRegression().fit(F, y)
                self.models[-1].append(model)
        self.models = list(reversed(self.models))

    def value(self, models_per_action, F):
        v = np.zeros((self.num_sims, self.num_actions))
        for i in range(self.num_actions):
            v[:, i] = models_per_action[i].predict(F)
        return v

    def action(self, states):
        i = states.time
        flat_correlation = states.correlations[:, self.flat_row, self.flat_col].T
        X = np.concatenate([states.spots, states.variances, flat_correlation, states.weights], axis=1)
        F = PolynomialFeatures(2).fit_transform(X)
        action_indices = np.argmax(self.value(self.models[i], F), axis=1)
        return action_indices

def q_approx_all(states, K, heston_params, T_rem, p_low, p_high):
    n, m = states.spots.shape
    #va = finance.valuatorAtm(heston_params, T_rem)
    dist = np.zeros((n, 1+m*(m-1)))
    for i in range(n):
        state = State(states.time, states.spots[i], states.variances[i], states.correlations[i], states.weights[i])
        actions, dist[i] = q_approx(state, K, heston_params, T_rem, p_low, p_high, None)
    return actions, dist


def q_approx(state, K, heston_params, T_rem, p_low, p_high, va):
    spots = state.spot
    V0s = state.variance
    correls = state.correlation
    w = state.weight

    B = np.dot(w, spots)
    n = heston_params.shape[0]
    vols = []
    for i in range(n):
        _, kappa, theta, alpha, rho = heston_params[i]
        price = va.value(i, V0s[i], spots[i]) if va else finance.value(spots[i], [spots[i]], V0s[i], kappa, theta, alpha, rho, T_rem)[0]
        q = -1  # put
        num_it = 1
        iv = py_lets_be_quickly_rational.implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
            price, spots[i], spots[i], T_rem, q, num_it)
        vols.append(iv)

    actions = [(0,0)] + [(i, j) for i in range(n) for j in range(n) if i != j]
    q_values = [finance.basket_price(spots, vols, correls, w, T_rem, K)]
    for i_inc, i_dec in actions[1:]:
        p_inc = w[i_inc] * spots[i_inc] / B
        p_dec = w[i_dec] * spots[i_dec] / B
        change = min(0.05, min(p_high - p_inc, p_dec - p_low))
        if change <= 0:
            q_values.append(q_values[0])
        else:
            w_inc_orig = w[i_inc]
            w_dec_orig = w[i_dec]
            w[i_inc] = (p_inc + change) * B / spots[i_inc]
            w[i_dec] = (p_dec - change) * B / spots[i_dec]
            q_values.append(finance.basket_price(spots, vols, correls, w, T_rem, K))
            w[i_inc] = w_inc_orig
            w[i_dec] = w_dec_orig

    return actions, np.array(q_values)

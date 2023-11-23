import py_lets_be_quickly_rational
import finance
from state import State, States
from env import Env

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
from copy import deepcopy
from collections import deque


class PolicyNoChange:
    def action(self, states):
        n_sim = states.weights.shape[0]
        return np.zeros(n_sim)


class PolicyRandom:
    def __init__(self, num_assets):
        self.num_actions = 1 + num_assets * (num_assets - 1)

    def action(self, states):
        num_sims = states.weights.shape[0]
        return np.random.choice(self.num_actions, size=num_sims)


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
        model.add(Dense(256, activation='relu', input_shape=(self.num_flat_state,)))
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


class PolicyLS:
    def __init__(self, policy, S, V, C, initial_weights, K, T):
        self.K = K
        self.T = T

        self.num_periods, self.num_sims, self.num_assets = S.shape
        self.num_actions = 1 + self.num_assets*(self.num_assets-1)
        self.states = []
        self.afterstate_weights = []
        weights = deepcopy(initial_weights)
        for i in range(self.num_periods):
            weights = weights * (S[i] if i==0 else S[i]/S[i-1])
            self.states.append(States(i, V[i], C[i], weights))
            if i < self.num_periods-1:
                action_indices = policy.action(self.states[-1])
                weights = Env.update_weight(weights, action_indices)
                self.afterstate_weights.append(weights)

        self.flat_row, self.flat_col = np.triu_indices(self.num_assets, k=1)
        self.models = []
        self.train()

    def train(self):
        for i in range(self.num_periods-2, -1, -1):
            flat_correlation = self.states[i].correlations[:, self.flat_row, self.flat_col]
            X = np.concatenate([self.states[i].variances**0.5, flat_correlation, self.afterstate_weights[i]], axis=1) # add t_rem?
            F = PolynomialFeatures(2).fit_transform(X)
            if i==self.num_periods-2:
                B = np.sum(self.states[i+1].weights, axis=1)
                y = np.maximum(B - self.K, 0)
            else:
                # TD learning
                ys = self.value(self.models[-1], self.states[i+1])
                y = np.max(ys, axis=1)
            # model = XGBRegressor(n_estimators=100, max_depth=5, eta=0.1, subsample=0.7, colsample_bytree=1.0)
            # model.fit(F, y)
            model = LinearRegression().fit(F, y)
            self.models.append(model)
        self.models = list(reversed(self.models))

    def value(self, next_model, next_states):
        num_sims = next_states.variances.shape[0]
        next_flat_correlation = next_states.correlations[:, self.flat_row, self.flat_col]
        ys = np.zeros((num_sims, self.num_actions))
        for j in range(self.num_actions):
            action_indices = j * np.ones(num_sims)
            next_afterstate_weights = Env.update_weight(next_states.weights, action_indices)
            next_X = np.concatenate([next_states.variances**0.5, next_flat_correlation, next_afterstate_weights],
                                    axis=1)
            next_F = PolynomialFeatures(2).fit_transform(next_X)
            ys[:, j] = next_model.predict(next_F)
        return ys

    def action(self, states):
        i = states.time
        action_indices = np.argmax(self.value(self.models[i], states), axis=1)
        return action_indices


def q_approx_all(states, K, heston_params, T_rem, p_low, p_high):
    n, m = states.weights.shape
    #va = finance.valuatorAtm(heston_params, T_rem)
    dist = np.zeros((n, 1+m*(m-1)))
    for i in range(n):
        state = State(states.time, states.variances[i], states.correlations[i], states.weights[i])
        actions, dist[i] = q_approx(state, K, heston_params, T_rem, p_low, p_high, None)
    return actions, dist


def q_approx(state, K, heston_params, T_rem, p_low, p_high, va):
    n = heston_params.shape[0]
    spot = 1 #we can assume that the current spot of every asset is 1
    V0s = state.variance
    correls = state.correlation
    w = state.weight

    B = np.sum(w)
    vols = []
    for i in range(n):
        _, kappa, theta, alpha, rho = heston_params[i]
        price = va.value(i, V0s[i], spot) if va else finance.value(spot, [spot], V0s[i], kappa, theta, alpha, rho, T_rem)[0]
        q = -1  # put
        num_it = 1
        iv = py_lets_be_quickly_rational.implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
            price, spot, spot, T_rem, q, num_it)
        vols.append(iv)

    actions = [(0,0)] + [(i, j) for i in range(n) for j in range(n) if i != j]
    q_values = [finance.basket_price(vols, correls, w, T_rem, K)]
    for i_inc, i_dec in actions[1:]:
        p_inc = w[i_inc] / B
        p_dec = w[i_dec] / B
        change = min(0.05, min(p_high - p_inc, p_dec - p_low))
        if change <= 0:
            q_values.append(q_values[0])
        else:
            w[i_inc] = (p_inc + change) * B
            w[i_dec] = (p_dec - change) * B
            q_values.append(finance.basket_price(vols, correls, w, T_rem, K))
            w[i_inc] = p_inc * B
            w[i_dec] = p_dec * B

    return actions, np.array(q_values)

import numpy as np
from copy import deepcopy


class StochEnv:

    def __init__(self, heston_params, spot_correlations, T, num_periods, num_sims):
        self.heston_params = heston_params # per row: V, meanReversionSpeed, longTermV, volOfVol, correlation
        self.spot_correlations = spot_correlations
        self.T = T
        self.num_periods = num_periods
        self.num_sims = num_sims

        self.num_periods_per_year = self.num_periods // self.T
        self.num_sim_steps_per_period = 365 // self.num_periods_per_year
        num_sim_steps_per_year = self.num_sim_steps_per_period * self.num_periods_per_year
        self.dt = 1.0 / num_sim_steps_per_year
        self.sqrtDt = self.dt**0.5
        self.na = self.heston_params.shape[0]
        self.U0 = np.linalg.cholesky(self.spot_correlations).T
        self.vol_of_correlation  = 0.002

    def simulate(self):
        Sr = np.zeros((self.num_periods, self.num_sims, self.na))
        Vr = np.zeros((self.num_periods, self.num_sims, self.na))
        Cr = np.ones((self.num_periods, self.num_sims, self.na, self.na))

        X = np.zeros(self.na)
        V = deepcopy(self.heston_params[:, 0])
        U = np.repeat(self.U0[np.newaxis, :, :], self.num_sims, axis=0)
        rho = np.ones((self.num_sims, self.na, self.num_sim_steps_per_period))
        for i in range(self.na):
            rho[:, i, :] = self.heston_params[i, -1]
        rho_bar = (1 - rho**2)**0.5

        for i in range(self.num_periods):
            U = U + np.random.multivariate_normal(np.zeros(self.na), self.vol_of_correlation**2*np.eye(self.na), size=(self.num_sims, self.na))
            U = U / np.linalg.norm(U, axis=1).reshape((self.num_sims, 1, self.na))
            L = np.transpose(U, axes=[0,2,1])
            W = np.random.normal(size=(self.num_sims, self.na, self.num_sim_steps_per_period))
            W1 = np.matmul(L, W)
            W2 = rho * W1 + rho_bar * np.random.normal(size=(self.num_sims, self.na, self.num_sim_steps_per_period))

            for j in range(self.num_sim_steps_per_period):
                sqrtV = np.sqrt(V)
                X = X + sqrtV * self.sqrtDt * W1[:, :, j] - 0.5 * V * self.dt
                V = np.abs(V + self.heston_params[:,1] * (self.heston_params[:, 2] - V) * self.dt +
                           self.heston_params[:,3] * sqrtV * self.sqrtDt * W2[:, :, j])

            Sr[i] = np.exp(X)
            Vr[i] = deepcopy(V)
            Cr[i] = np.matmul(L, U)

        return Sr, Vr, Cr


class Env:
    p_low = 0.1
    p_high = 0.5

    def __init__(self):
        pass

    @staticmethod
    def update_weight(weights, actions_indices):
        num_sims, num_assets = weights.shape
        pairs = [(i,j) for i in range(num_assets) for j in range(num_assets) if i!=j]
        action_decoder = {i+1: pairs[i] for i in range(len(pairs))}
        action_decoder[0] = (0, 0)
        updated_weights = deepcopy(weights)
        for i in range(num_sims):
            w = updated_weights[i]
            B = np.sum(w)

            i_inc, i_dec = action_decoder[actions_indices[i]]
            if i_inc!=0 or i_dec!=0:
                p_inc = w[i_inc] / B
                p_dec = w[i_dec] / B
                change = min(0.05, min(Env.p_high - p_inc, p_dec - Env.p_low))
                w[i_inc] = (p_inc + change) * B
                w[i_dec] = (p_dec - change) * B
        return updated_weights

    @staticmethod
    def compute_payoff(states, K):
        B = np.sum(states.weights, axis=1)
        payoff = np.maximum(B - K, 0)
        sd = np.std(payoff) / len(B)**0.5
        return np.mean(payoff) - 2*sd, np.mean(payoff) + 2*sd

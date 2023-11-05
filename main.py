from env import StochEnv, Env
from agents import PolicyGreedy, PolicyRandom, PolicyLS, PolicyNoChange
from state import States

import numpy as np
import finance

import time


if __name__ == '__main__':
    heston_params = np.array([
        [0.25**2, 3.0, 0.30**2, 0.4, -0.6],
        [0.35**2, 3.0, 0.25**2, 0.4, -0.5],
        [0.30**2, 2.0, 0.30**2, 0.3, -0.7],
        [0.40**2, 2.0, 0.40**2, 0.3, -0.4]
    ])
    spot_correlations = np.array([
        [1.0, 0.7, 0.4, 0.3],
        [0.7, 1.0, 0.5, 0.3],
        [0.4, 0.5, 1.0, 0.3],
        [0.3, 0.3, 0.3, 1.0]
    ])

    num_assets = heston_params.shape[0]
    num_sims = 10000
    num_periods = 8
    T=2

    start = time.time()
    se = StochEnv(heston_params, spot_correlations, T, num_periods, num_sims)
    S, V, C = se.simulate()

    K = 1
    weights = (1/num_assets)*np.ones((num_sims, num_assets)) # initial afterstate

    # define a policy
    # policy_train = PolicyRandom(num_assets)
    # S_train, V_train, C_train = se.simulate()
    # policy = PolicyLS(policy_train, S_train, V_train, C_train, weights, K, T)
    policy = PolicyGreedy(K, heston_params, Env.p_low, Env.p_high, T, num_periods)
    # policy = PolicyRandom(num_assets)
    # policy = PolicyNoChange()

    # evaluate the policy
    for i in range(num_periods):
        # at the end of period i
        weights = weights * (S[i] / S[i-1] if i>0 else S[i])  # new state
        states = States(i, V[i], C[i], weights) # contains the state for all simulations (ie num_sims)
        if i+1<num_periods:
            action_indices = policy.action(states)
            weights = Env.update_weight(weights, action_indices) # afterstate
        else:
            print('payoff:', Env.compute_payoff(states, K))
    end = time.time()
    print("time:", (end - start) * 10 ** 3, "ms")

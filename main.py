from env import StochEnv, Env
from agents import PolicyGreedy, PolicyRandom, PolicyLS
from state import States

import numpy as np
import finance

import time


if __name__ == '__main__':
    heston_params = np.array([
        [0.25**2, 3.0, 0.3**2, 0.4, -0.6],
        [0.35**2, 3.0, 0.25**2, 0.4, -0.5],
        [0.3**2, 2.0, 0.3**2, 0.3, -0.7],
        [0.4 ** 2, 2.0, 0.4 ** 2, 0.3, -0.4]
    ])
    spot_correlations = np.array([
        [1.0, 0.7, 0.4, 0.3],
        [0.7, 1.0, 0.5, 0.3],
        [0.4, 0.5, 1.0, 0.3],
        [0.3, 0.3, 0.3, 1.0]
    ])

    num_assets = heston_params.shape[0]
    num_sims = 20000
    num_periods= 4
    T=1

    start = time.time()
    se = StochEnv(heston_params, spot_correlations, T, num_periods, num_sims)
    S, V, C = se.simulate()

    K = 1
    weights = np.repeat((1/num_assets)*np.ones(num_assets).reshape((1,-1)), num_sims, axis=0)

    # define a policy
    policy_train = PolicyRandom(num_assets)
    S_train, V_train, C_train = se.simulate()
    policy = PolicyLS(policy_train, S_train, V_train, C_train, weights, K, T)
    #policy = PolicyGreedy(K, heston_params, Env.p_low, Env.p_high, T, num_periods)

    # evaluate the policy
    for i in range(num_periods):
        # at the end of period i
        states = States(i, S[i], V[i], C[i], weights) # contains the state for all simulations (ie num_sims)
        if i+1<num_periods:
            action_indices = policy.action(states)
            weights = Env.update_weight(states, action_indices, num_sims, num_assets)
        else:
            print('payoff:', Env.compute_payoff(states, K))
    end = time.time()
    print("time:", (end - start) * 10 ** 3, "ms")

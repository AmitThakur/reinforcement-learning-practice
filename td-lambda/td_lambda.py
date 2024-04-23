import numpy as np
from utils.common_utils import decay_schedule
from tqdm import tqdm


def td_lambda(pi, env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
              lamda=0.3, n_episodes=500):
    """
    Predict the value function using Temporal Difference (Lambda) algorithm (backward view TD(lambda)).
    TD(0): TD, TD(1): MC
    :param pi:
    :param env:
    :param gamma:
    :param init_alpha:
    :param min_alpha:
    :param alpha_decay_ratio:
    :param lamda:
    :param n_episodes:
    :return:
    """
    # No of possible states
    n_space = env.observation_space.n
    # Initialize value function to zero
    V = np.zeros(n_space, dtype=np.float64)
    # Initialize Eligibility traces to zero
    E = np.zeros(n_space, dtype=np.float64)
    # Generate decaying alpha sequence
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    # Run through the episodes
    for e in tqdm(range(n_episodes), leave=False):
        # Initialize Eligibility traces to zero
        E.fill(0)
        # Start a new episode
        state, done = env.reset(), False
        while not done:
            action = pi(state)
            next_state, reward, done, _ = env.step(action)
            td_target = reward + gamma * V[next_state] * (not done)
            td_error = td_target - V[state]
            E[state] = E[state] + 1
            # Update the complete value function with each step
            V = V + alphas[e] * td_error * E
            # Decay the eligibility traces
            E = gamma * lamda * E
            state = next_state
    return V
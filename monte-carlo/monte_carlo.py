from utils.common_utils import generate_trajectory, decay_schedule
import numpy as np
from tqdm import tqdm


def mc_prediction(pi, env, gamma=1.0, init_alpha=0.5, min_alpha=0.01,
                  alpha_decay_ratio=0.5, n_episodes=500, max_steps=200,
                  first_visit=True):
    """
    Predict value function using Monte Carlo algorithm
    :param pi: policy
    :param env: Environment
    :param gamma: discount factor
    :param init_alpha: Initial learning rate
    :param min_alpha: Decayed learning rate
    :param alpha_decay_ratio: In what fraction of steps should decay be completed.
    :param n_episodes: No of episodes
    :param max_steps: Max steps allowed within an episode
    :param first_visit: Is First Visit rule applicable? True/False
    :return: Estimated Value Function using Monte Carlo algorithm
    """
    # No of possible states
    n_state = env.observation_space.n

    # Pre calculate discount factor sequence
    discounts = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)

    # Pre-decaying alpha sequence: More Exploring initially, more exploiting later
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)

    # Initialize the value function to zero
    V = np.zeros(n_state, dtype=np.float64)

    # Run multiple episodes
    for e in tqdm(range(n_episodes), leave=False):
        # Generate a trajectory
        trajectory = generate_trajectory(pi, env, max_steps)
        # Initialize a lookup for visited status for state
        visited = np.zeros(n_state, dtype=np.dtype(bool))
        for t, (state, _, reward, _, _) in enumerate(trajectory):
            if visited[state] and first_visit:
                continue  # If FV rule is applicable, then we skip for second visit onward
            visited[state] = True

            # Calculate the length of steps from current to end [t, end]
            n_steps = len(trajectory[t:])
            # MC Target: G
            G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
            # MC Error
            mc_error = G - V[state]
            # Using decayed learning rate: alpha
            V[state] = V[state] + alphas[e] * mc_error

    return V.copy()

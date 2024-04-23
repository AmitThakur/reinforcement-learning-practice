from utils.common_utils import (generate_trajectory,
                                generate_trajectory_epsilon_greedy, decay_schedule)
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
            # MC Target: G which is an actual return calculated till end of the trajectory
            G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
            # MC Error
            mc_error = G - V[state]
            # Update V using actual returns till end of the episode
            V[state] = V[state] + alphas[e] * mc_error

    return V.copy()


def mc_control(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
               init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9,
               n_episodes=3000, max_steps=200, first_visit=True):
    """
    Get the improved action-value-function, value-function, and policy using Monte Carlo algorithm
    :param env: Environment
    :param gamma: Discount factor
    :param init_alpha: Initial learning rate
    :param min_alpha: Decayed learning rate
    :param alpha_decay_ratio: In what fraction of steps should decay be completed.
    :param init_epsilon: Initial exploration probability
    :param min_epsilon: Decayed exploration probability
    :param epsilon_decay_ratio:
    :param n_episodes: No of episodes
    :param max_steps: Max steps allowed within an episode
    :param first_visit: Is First Visit rule applicable? True/False
    :return: improved action-value-function, value-function, and policy using Monte Carlo algorithm
    """
    n_state, n_action = env.observation_space.n, env.action_space.n
    discounts = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)
    Q = np.zeros((n_state, n_action), dtype=np.float64)
    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
        if np.random.random() > epsilon \
        else np.random.randint(len(Q[state]))

    for e in tqdm(range(n_episodes), leave=False):
        trajectory = generate_trajectory_epsilon_greedy(select_action, Q, epsilons[e], env, max_steps)
        visited = np.zeros((n_state, n_action), dtype=np.dtype(bool))
        for t, (state, action, reward, _, _) in enumerate(trajectory):
            if visited[state][action] and first_visit:
                continue
            visited[state][action] = True

            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
            Q[state][action] = Q[state][action] + alphas[e] * (G - Q[state][action])

    V = np.max(Q, axis=1)
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return Q, V, pi

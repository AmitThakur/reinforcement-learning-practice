from utils.common_utils import decay_schedule
import numpy as np
from tqdm import tqdm


def td_prediction(pi, env, gamma=1.0, init_alpha=0.5, min_alpha=0.01,
                  alpha_decay_ratio=0.5, n_episodes=500):
    """
    Predict value function using temporal difference algorithm
    :param pi: policy
    :param env: Environment
    :param gamma: discount factor
    :param init_alpha:  Initial learning rate
    :param min_alpha: Fully decayed learning rate
    :param alpha_decay_ratio:  In what fraction of steps should decay be completed.
    :param n_episodes: No of episodes
    :return: Estimated Value Function using Temporal Difference algorithm
    """
    # No of possible states
    n_state = env.observation_space.n

    # Initialize the value function to zero
    V = np.zeros(n_state, dtype=np.float64)

    # Pre-decaying alpha sequence: More Exploring initially, more exploiting later
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)

    # Run multiple episodes
    for e in tqdm(range(n_episodes), leave=False):
        # Start an episode
        state, done = env.reset(), False
        while not done:
            # Select an action according to the policy
            action = pi(state)
            # Take step and observe the transition
            next_state, reward, done, _ = env.step(action)
            td_target = reward + gamma * V[next_state] * (not done)
            td_error = td_target - V[state]
            # Unlike MC methods, we update V for current state per
            # step using current estimate of next state: using estimate to update estimates
            # "reward" injects reality into the estimates, one step at a time.
            # No need to traverse the trajectory till end like MC to get target return
            V[state] = V[state] + alphas[e] * td_error
            state = next_state
    return V

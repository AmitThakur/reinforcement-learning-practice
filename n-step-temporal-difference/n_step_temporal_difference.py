import numpy as np
from utils.common_utils import decay_schedule
from tqdm import tqdm


def n_step_temporal_difference(pi, env, gamma=1.0, init_alpha=0.5, min_alpha=0.01,
                               alpha_decay_ratio=0.5, n_step=3, n_episodes=500):
    """
    Predict value function using temporal difference algorithm
    :param pi: policy
    :param env: Environment
    :param gamma: discount factor
    :param init_alpha:  Initial learning rate
    :param min_alpha: Fully decayed learning rate
    :param alpha_decay_ratio:  In what fraction of steps should decay be completed.
    :param n_step: No of steps (n) after which Value function should be updated
    :param n_episodes: No of episodes
    :return: Estimated Value Function using Temporal Difference algorithm
    """
    n_state = env.observation_space.n
    V = np.zeros(n_state, dtype=np.float64)
    # n_step discounts
    discounts = np.logspace(0, n_step+1, num=n_step+1, base=gamma, endpoint=False)
    # decaying alphas
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    for e in tqdm(range(n_episodes), leave=False):
        state, done, path = env.reset(), False, []
        # Path to hold recent n step experiences
        while not done or path is not None:
            # pop the first element of the path
            path = path[1:]
            next_state = None
            while not done and len(path) < n_step:
                action = pi(state)
                next_state, reward, done, _ = env.step(action)
                experience = (state, reward, next_state, done)
                path.append(experience)
                state = next_state
                if done:
                    break

            n = len(path)
            est_state = path[0][0]
            rewards = np.array(path)[:, 1]
            partial_return = discounts[:n] * rewards

            bootstrap_val = discounts[-1] * V[next_state] * (not done)
            ntd_target = np.sum(np.append(partial_return, bootstrap_val))
            ntd_error = ntd_target - V[est_state]
            V[est_state] = V[est_state] + alphas[e] * ntd_error
            # If the only experience is of terminal state, then break the loop by setting path to none
            if len(path) == 1 and path[0][3]:
                path = None
    return V

import numpy as np
from utils.common_utils import decay_schedule, choose_epsilon_greedy_action
from tqdm import tqdm


def q_learning(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5, init_epsilon=1.0,
          min_epsilon=0.1, epsilon_decay_ratio=0.9, n_episodes=3000):
    """
    Solve Control Problem using Q-Learning.

    :param env: Environment
    :param gamma: discount factor
    :param init_alpha: Initial learning rate
    :param min_alpha: Fully decayed learning rate
    :param alpha_decay_ratio: In what fraction of steps should decay be completed.
    :param init_epsilon: Initial Exploring Rate
    :param min_epsilon: Fully decayed Fully decayed
    :param epsilon_decay_ratio: In what fraction of steps should epsilon decay be completed.
    :param n_episodes: No of episodes
    :return: Estimated functions: Q (Action Value), V(State Value), pi (Policy)
    """
    # Get the spec sizes
    n_states, n_actions = env.observation_space.n, env.action_space.n

    # Initialize Q value function to zero.
    Q = np.zeros((n_states, n_actions), dtype=np.float64)

    # Pre-decaying alpha sequence: Learning Rate
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    # Pre-decaying epsilon sequence: Exploring Rate
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    # Run through the episodes
    for e in tqdm(range(n_episodes), leave=False):
        # Initialize the env
        state, done = env.reset(), False
        # We don't compute trajectory here upfront like MC methods. We use TD.
        while not done:
            # Choose an epsilon-greedy action with decaying exploration rate.
            action = choose_epsilon_greedy_action(state, Q, epsilons[e])
            # Interact with the env till end
            next_state, reward, done, _ = env.step(action)
            # Compute TD target, which uses Q estimates of next state-(but max valued) action pair.
            td_target = reward + gamma * Q[next_state].max() * (not done)
            # Compute TD error
            td_error = td_target - Q[state][action]
            # Q estimate update
            Q[state][action] = Q[state][action] + alphas[e] * td_error
            state = next_state

    # Greedily choose value function: Max Q value for each state.
    V = np.max(Q, axis=1)
    # Greedily choose policy: Max Q valued action for each state.
    pi = lambda s: {s :a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return Q, V, pi
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import cycle, count
from prettytable import PrettyTable

np.set_printoptions(suppress=True)

random.seed(123)
np.random.seed(123)


def plot_policy(pi, P, n_rows, n_cols, title, action_symbols, start_state, goal_state):
    """
    Plot a policy
    :param pi: policy
    :param P: MDP
    :param n_rows: No of rows
    :param n_cols: No of columns
    :param title: Title of plot
    :param action_symbols: An array of action symbols corresponding to their index
    :param start_state: Start state
    :param goal_state: Goal State
    :return: Plot the policy.
    """
    n_states = len(P)
    assert (n_rows * n_cols) == n_states
    fig, ax = plt.subplots(figsize=(n_cols * 1.5, n_rows * 1.5))
    plt.gca().invert_yaxis()
    plt.subplots_adjust(top=0.60)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,
        right=False,
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,  # labels along the bottom edge are off
        labelleft=False
    )
    ax.grid(True)
    ax.set_xticks(np.arange(0, n_cols + 1, 1.0))
    ax.set_yticks(np.arange(0, n_rows + 1, 1.0))
    plt.title(title)

    for y in range(n_rows):
        for x in range(n_cols):
            state_index = int(x + (y * n_cols))
            if state_index == start_state:
                cell_val = 'S'
            elif state_index == goal_state:
                cell_val = 'G'
            else:
                cell_val = action_symbols[pi(state_index)]

            ax.text(x + 0.4, y + 0.5, f'{cell_val}', color='blue')
            ax.text(x + 0.08, y + 0.2, f'{state_index}', color='gray')

    plt.show()


def plot_state_value_function(n_rows, n_cols, V, title, precision=5):
    """
    Plot state value function (V)
    :param n_rows: No of rows
    :param n_cols: No of cols
    :param V: State Value function
    :param title: Title of the plot
    :param precision: Precision of decimal numbers
    :return: Plot the V.
    """
    n_states = len(V)
    assert (n_rows * n_cols) == n_states
    fig, ax = plt.subplots(figsize=(n_cols * 1.5, n_rows * 1.5))
    plt.gca().invert_yaxis()
    plt.subplots_adjust(top=0.60)
    plt.tick_params(
        axis='both',
        which='both',
        left=False,
        right=False,
        bottom=False,
        top=False,
        labelbottom=False,
        labelleft=False
    )
    ax.grid(True)
    ax.set_xticks(np.arange(0, n_cols + 1, 1.0))
    ax.set_yticks(np.arange(0, n_rows + 1, 1.0))
    plt.title(title)

    for y in range(n_rows):
        for x in range(n_cols):
            state = int(x + (y * n_cols))
            cell_val = np.round(V[state], precision)
            ax.text(x + 0.4, y + 0.5, f'{cell_val}', color='blue')
            ax.text(x + 0.08, y + 0.2, f'{state}', color='gray')

    plt.show()


def evaluate_policy(pi, P, gamma=1.0, theta=1e-10):
    """
    Calculate the value function using MDP
    :param pi: policy
    :param P: MDP
    :param gamma: discount rate
    :param theta: threshold of change in Value function, below which the evaluation should stop
    :return: Value function (V)
    """
    prev_V = np.zeros(len(P), dtype=np.float64)
    while True:
        V = np.zeros(len(P), dtype=np.float64)
        for state in range(len(P)):
            action = pi(state)
            for prob, next_state, reward, done in P[state][action]:
                V[state] += prob * (reward + gamma * prev_V[next_state] * (not done))
        # Condition for convergence
        if np.max(np.abs(prev_V - V)) < theta:
            break
        prev_V = V.copy()
    return V


def improve_policy(V, P, gamma=1.0):
    """
    Improve the policy using MDP (P) and given value function (V)
    :param V: Policy
    :param P: MDP
    :param gamma: Discount factor
    :return: Improved policy (pi')
    """
    # Initialize Action Value Function to zero
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])):
            # We are using MDP, not directly interacting with the env
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

    new_pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return new_pi


def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    """
    Calculate probability of success that following the policy we will reach the goal
    :param env:
    :param pi:
    :param goal_state:
    :param n_episodes:
    :param max_steps:
    :return:
    """
    random.seed(123)
    np.random.seed(123)
    results = []
    for _ in range(n_episodes):
        # State of a new episode
        state, done, steps = env.reset(), False, 0
        # Keep interacting with the env till we reach the terminal state
        # or exceed max_step limit
        while not done and steps < max_steps:
            # We are ignoring the reward value here
            state, _, done, _ = env.step(pi(state))
            # print(env.step(pi(state)))
            # state, reward, terminated, truncated, info = env.step(pi(state))
            steps += 1

        # only interested in knowing whether we reached the goal or not
        results.append(state == goal_state)
    return np.sum(results) / len(results)


def mean_return(env, pi, n_episodes=100, max_steps=200, gamma=1.0):
    """
    Calculate mean return of all episodes to reach a
    :param env: Environment
    :param pi: Policy
    :param n_episodes: No of episodes
    :param max_steps:
    :param gamma: Discount factor
    :return: Mean return
    """
    random.seed(123)
    np.random.seed(123)
    # env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += (gamma**steps * reward)
            steps += 1
    return np.mean(results)


def print_policy_success_stats(env, pi, goal_state, n_episodes=100, max_steps=200, gamma=1.0):
    success_percent = probability_success(env, pi, goal_state, n_episodes, max_steps)
    return_mean = mean_return(env, pi, n_episodes, max_steps, gamma)
    print("By using the Policy, success rate to reach goal is: {:.2f}%".format(success_percent * 100))
    print("By using the Policy, mean reward is: {:.4f}".format(return_mean))


def generate_random_policy(P):
    """
    Generate a random policy
    :param P: MDP
    :return: Generated random policy
    """
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    pi = lambda s: {s: a for s, a in enumerate(random_actions)}[s]
    return pi


def rmse(x, y, dp=4):
    """
    Calculate root mean square error
    :param x: First variable
    :param y: Second variable
    :param dp: Decimal Places
    :return: RMSE value
    """
    return np.round(np.sqrt(np.mean((x - y)**2)), dp)


def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
    """
    Generate the decay schedule. More Exploring initially, more exploiting later.
    :param init_value: Initial value
    :param min_value: Min value
    :param decay_ratio: Decay Ration: In what fraction of steps should decay be completed.
    :param max_steps: Max steps
    :param log_start: Starting exponent
    :param log_base: Log base
    :return: The generated decay schedule
    """
    # We'll take decay_steps to decay from init_value down to min_value
    decay_steps = int(max_steps * decay_ratio)
    # Leftover steps
    rem_steps = max_steps - decay_steps
    # Create a sequence [reversed] of scaling numbers from base^log_start to base^0 (=1), including end points
    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    # Normalize them (min-max scaling)
    values = (values - values.min()) / (values.max() - values.min())
    # Get the scaled values
    values = (init_value - min_value) * values + min_value
    # Pad the sequence with edge numbers to each size with width left(0) and right(rem_steps)
    values = np.pad(values, (0, rem_steps), 'edge')
    return values


def generate_trajectory(pi, env, max_steps=200):
    """
    Generate a trajectory (episode) of interaction with env and state transition
    :param pi: Policy we use
    :param env: Environment instance
    :param max_steps: Maximum steps limit
    :return: The generated trajectory
    """
    done, trajectory = False, []
    while not done:
        state = env.reset()
        for t in count():
            action = pi(state)
            next_state, reward, done, _ = env.step(action)
            experience = (state, action, reward, next_state, done)
            trajectory.append(experience)
            if done:
                break
            if t >= max_steps - 1:
                # Discarding the trajectory that didn't end within max_steps.
                trajectory = []
                break
            state = next_state

    return np.array(trajectory, np.dtype(object))


def generate_trajectory_epsilon_greedy(select_action, Q, epsilon, env, max_steps=200):
    """
    Generate a trajectory (episode) of interaction with env and state transition
    :param select_action: Action selection function
    :param Q: Action Value function
    :param epsilon: Probability of exploration
    :param env: Environment instance
    :param max_steps: Maximum steps limit
    :return: The generated trajectory
    """
    done, trajectory = False, []
    while not done:
        # New episode
        state = env.reset()
        for t in count():
            action = select_action(state, Q, epsilon)
            next_state, reward, done, _ = env.step(action)
            experience = (state, action, reward, next_state, done)
            trajectory.append(experience)
            if done:
                break
            if t >= max_steps - 1:
                # Discarding the trajectory that didn't end within max_steps.
                trajectory = []
                break
            state = next_state

    return np.array(trajectory, np.dtype(object))


def print_action_value_function(Q, action_symbols):
    """
    Prints the action value function Q
    :param Q: action value function
    :param action_symbols: Action Symbols array
    :return: Prints in tabular format
    """
    table = PrettyTable()
    table.field_names = ['state'] + action_symbols
    for state, entry in enumerate(Q):
        table.add_row(np.concatenate(([state], entry)))
    print(table)


def get_policy_metrics(env, gamma, pi, goal_state, optimal_Q, n_episodes=100, max_steps=200):
    """
    Get policy metrics:
    :param env:
    :param gamma:
    :param pi:
    :param goal_state:
    :param optimal_Q:
    :param n_episodes:
    :param max_steps:
    :return:
    """
    random.seed(123)
    np.random.seed(123)
    reached_goal, episode_reward, episode_regret = [], [], []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        episode_reward.append(0.0)
        episode_regret.append(0.0)
        while not done and steps < max_steps:
            action = pi(state)
            regret = np.max(optimal_Q[state]) - optimal_Q[state][action]
            episode_regret[-1] += regret

            state, reward, done, _ = env.step(action)
            episode_reward[-1] += (gamma ** steps * reward)

            steps += 1

        reached_goal.append(state == goal_state)
    results = np.array((np.sum(reached_goal) / len(reached_goal) * 100,
                        np.mean(episode_reward),
                        np.mean(episode_regret)))
    return results


def moving_average(a, n=100):
    """
    Calculate moving average
    :param a: array
    :param n: limit
    :return: Moving average
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
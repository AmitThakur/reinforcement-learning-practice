import numpy as np
import matplotlib.pyplot as plt
import random

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
    return np.mean(results)


def mean_return(env, pi, n_episodes=100, max_steps=200):
    """
    Calculate mean return of all episodes to reach a
    :param env:
    :param pi:
    :param n_episodes:
    :param max_steps:
    :return:
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
            results[-1] += reward
            steps += 1
    return np.mean(results)


def print_policy_success_stats(env, pi, goal_state, n_episodes=100, max_steps=200):
    success_percent = probability_success(env, pi, goal_state, n_episodes, max_steps)
    return_mean = mean_return(env, pi, n_episodes, max_steps)
    print("By using the Policy, success rate to reach goal is: {:.2f}%".format(success_percent * 100))
    print("By using the Policy, un-discounted reward is: {:.4f}".format(return_mean))


def generate_random_policy(P):
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    pi = lambda s: {s: a for s, a in enumerate(random_actions)}[s]
    return pi

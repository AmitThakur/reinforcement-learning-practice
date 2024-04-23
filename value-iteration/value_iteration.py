import numpy as np
from utils.common_utils import evaluate_policy, improve_policy, generate_random_policy


def value_iteration(P, gamma=1.0, theta=1e-10):
    """
    Run value iteration using action-value function and greedy approach to get an optimal policy
    :param P: MDP
    :param gamma: discount factor
    :param theta: Threshold for value function evaluation
    :return: V, pi

    Usage: V,pi = value_iteration(V, P, gamma=0.99, theta=1e-10)
    """
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    # V is a old truncated estimate
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

        # Check for convergence of the Value function after each state space sweep
        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break

        # Value iteration: Greedy-Max Action value, based on current evaluated Value function
        # But in Policy iteration: the approximation is based on V[state] swaps
        V = np.max(Q, axis=1)

    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return V, pi
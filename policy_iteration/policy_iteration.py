from utils.common_utils import evaluate_policy, improve_policy, generate_random_policy


def policy_iteration(pi, P, gamma=1.0, theta=1e-10):
    """
    Run policy iteration (evaluation -> improvement) till policy converges
    :param P: MDP
    :param gamma: discount factor
    :param theta: Threshold for policy evaluation
    :return: V, pi

    Usage: V, pi = policy_iteration(P, gamma=0.99, theta=1e-10)
    """
    count = 0
    while True:
        # lambda to dictionary
        old_pi = {s: pi(s) for s in range(len(P))}
        V = evaluate_policy(pi, P, gamma, theta)
        pi = improve_policy(V, P, gamma)
        count += 1
        if old_pi == {s: pi(s) for s in range(len(P))}:
            print(f"Policy improved in {count} iterations")
            break
    return V, pi




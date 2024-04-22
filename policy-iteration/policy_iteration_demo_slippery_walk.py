import gym, gym_walk
from utils.policy_utils import plot_policy, evaluate_policy, plot_state_value_function, \
    print_policy_success_stats
from policy_iteration import policy_iteration

# Initialize an environment
env = gym.make('SlipperyWalkFive-v0')

# Capture its MDP
P = env.env.P

# Reset env
init_state = env.reset(seed=123)
# Set goal
goal_state = 6

# Define actions
LEFT, RIGHT = range(2)

# Define extremely adverse policy
pi = lambda s: {
    0: LEFT, 1: LEFT, 2: LEFT, 3:LEFT, 4: LEFT, 5: LEFT, 6: LEFT
}[s]

# Evaluate initial policy
V = evaluate_policy(pi, P)
plot_policy(pi, P, 1,7, "Initial Policy",
            ['<', '>'], init_state, goal_state)
print_policy_success_stats(env, pi, goal_state)
plot_state_value_function(1, 7, V, "Initial Value Function (from MDP)")

# Run policy iteration
V, pi = policy_iteration(pi, P)

# Print new policy and value function
plot_policy(pi, P, 1,7, "Improved Policy", ['<', '>'], init_state, goal_state)
print_policy_success_stats(env, pi, goal_state)
plot_state_value_function(1, 7, V, "Improved Value Function (from MDP)")




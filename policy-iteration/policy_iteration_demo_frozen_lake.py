import gym
from utils.policy_utils import plot_policy, evaluate_policy, plot_state_value_function, \
    print_policy_success_stats
from policy_iteration import policy_iteration

# Initialize an environment
env = gym.make('FrozenLake-v1')

# Capture its MDP
P = env.env.P

# Reset env
init_state = env.reset(seed=123)
# Set goal
goal_state = 15

# Define actions
LEFT, DOWN, RIGHT, UP = range(4)
action_symbols = ['<', 'v', '>', '^']

# Define random policy
pi = lambda s: {
    0:UP, 1:UP, 2:UP, 3:UP,
    4:UP, 5:LEFT, 6:UP, 7:LEFT,
    8:LEFT, 9:LEFT, 10:LEFT, 11:LEFT,
    12:LEFT, 13:LEFT, 14:LEFT, 15:LEFT
}[s]

rows, cols = 4, 4

# Evaluate initial policy
V = evaluate_policy(pi, P, gamma=0.99)
plot_policy(pi, P, rows, cols, "Initial Policy", action_symbols, init_state, goal_state)
print_policy_success_stats(env, pi, goal_state)
plot_state_value_function(rows, cols, V, "Initial Value Function (from MDP)")

# Run policy iteration
V, pi = policy_iteration(pi, P, gamma=0.99)

# Print new policy and value function
plot_policy(pi, P, rows, cols, "Improved Policy", action_symbols, init_state, goal_state)
print_policy_success_stats(env, pi, goal_state)
plot_state_value_function(rows, cols, V, "Improved Value Function (from MDP)")




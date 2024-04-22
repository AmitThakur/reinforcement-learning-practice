import gym, gym_walk
from utils.policy_utils import plot_policy, evaluate_policy, plot_state_value_function, \
    print_policy_success_stats
from value_iteration import value_iteration

# Initialize an environment
env = gym.make('SlipperyWalkFive-v0')

# Capture its MDP
P = env.env.P

# Reset env
init_state = env.reset(seed=123)
# Set goal
goal_state = 6
row, col = 1, 7

# Define actions
LEFT, RIGHT = range(2)
action_symbols = ['<', '>']

# Run value iteration
optimal_V, optimal_pi = value_iteration(P)
plot_policy(optimal_pi, P, row, col, "Optimal Policy", action_symbols, init_state, goal_state)
print_policy_success_stats(env, optimal_pi, goal_state)
plot_state_value_function(row, col, optimal_V, "Optimal Value Function")





import warnings
import gym, gym_walk, gym_aima
from value_iteration.value_iteration import value_iteration
from utils.common_utils import (plot_state_value_function, print_action_value_function,
                                get_policy_metrics, rmse)
import numpy as np
from monte_carlo import mc_control
warnings.filterwarnings('ignore')

env = gym.make('SlipperyWalkSeven-v0')
init_state = env.reset()
goal_state = 8
gamma = 0.99
n_episodes = 3000
P = env.env.P
row, col = 1, 9
action_symbols = ['<', '>']
optimal_Q, optimal_V, optimal_pi = value_iteration(P, gamma=gamma)
plot_state_value_function(row, col, optimal_V, 'Optimal State Value Function')

print_action_value_function(optimal_Q, action_symbols)
success_rate_op, mean_return_op, mean_regret_op = get_policy_metrics(
    env, gamma=gamma, pi=optimal_pi, goal_state=goal_state, optimal_Q=optimal_Q)
print('Success rate {:.2f}%. Average return of {:.4f}. Regret of {:.4f}'.format(
    success_rate_op, mean_return_op, mean_regret_op))


# Q_mc, V_mc, pi_mc = mc_control(env, gamma=gamma, n_episodes=n_episodes)
# plot_state_value_function(row, col, V_mc, 'MC-Control State Value Function')
# print('State-value function RMSE: {}'.format(rmse(V_mc, optimal_V)))
# print_action_value_function(Q_mc, action_symbols)
# success_rate_mc, mean_return_mc, mean_regret_mc = get_policy_metrics(
#     env, gamma=gamma, pi=pi_mc, goal_state=goal_state, optimal_Q=optimal_Q)
# print('Success rate {:.2f}%. Average return of {:.4f}. Regret of {:.4f}'.format(
#     success_rate_mc, mean_return_mc, mean_regret_mc))



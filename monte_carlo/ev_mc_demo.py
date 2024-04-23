import warnings
import gym, gym_walk
from monte_carlo import mc_prediction
from utils.common_utils import (evaluate_policy, print_policy_success_stats,
                                plot_state_value_function, plot_policy, rmse)
warnings.filterwarnings('ignore')

env = gym.make('RandomWalk-v0')
init_state = env.reset()
goal_state = 6
gamma = 1.0  # Episodic
n_episodes = 500
P = env.env.P

LEFT, RIGHT = range(2)
pi = lambda s: {
    0: LEFT, 1: LEFT, 2: LEFT, 3: LEFT, 4: LEFT, 5: LEFT, 6: LEFT
}[s]
action_symbols = ['<', '>']
row, col = 1, 7

V_true = evaluate_policy(pi, P, gamma=gamma)
plot_state_value_function(row, col, V_true, 'True Value Function')
plot_policy(pi, P, row, col, 'Initial Policy', action_symbols, init_state, goal_state)
print_policy_success_stats(env, pi, goal_state=goal_state, gamma=gamma)

V_fv_mc = mc_prediction(pi, env, gamma=gamma, n_episodes=n_episodes, first_visit=False)
plot_state_value_function(row, col, V_fv_mc, 'EV-MC Value Function')
plot_state_value_function(row, col, V_fv_mc - V_true, 'Error EV_MC Value Function')

print('RMSE:', rmse(V_fv_mc, V_true))






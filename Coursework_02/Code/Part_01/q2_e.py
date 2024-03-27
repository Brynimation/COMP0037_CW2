#!/usr/bin/env python3

'''
Created on 9 Mar 2023

@author: ucacsjj
'''

import math

from common.scenarios import corridor_scenario

from common.airport_map_drawer import AirportMapDrawer


from td.q_learner import QLearner

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    airport_map, drawer_height = corridor_scenario()

    # Show the scenario        
    airport_map_drawer = AirportMapDrawer(airport_map, drawer_height)
    airport_map_drawer.update()
    
    # Create the environment
    env = LowLevelEnvironment(airport_map)

    # Extract the initial policy. This is e-greedy
    pi = env.initial_policy()
    
    # Select the controller
    policy_learner = QLearner(env)   
    policy_learner.set_initial_policy(pi)

    # These values worked okay for me.
    policy_learner.set_alpha(0.1)
    policy_learner.set_experience_replay_buffer_size(64)
    policy_learner.set_number_of_episodes(32)
    
    # The drawers for the state value and the policy
    value_function_drawer = ValueFunctionDrawer(policy_learner.value_function(), drawer_height)    
    greedy_optimal_policy_drawer = LowLevelPolicyDrawer(policy_learner.policy(), drawer_height)
    
    episode_durations = []

    # Different empirical experiments
    # lst = [0.05, 0.95] * 40
    # pi.set_epsilon(1)
    # for i in range(40, -1, -1):
    for i in range(40):

        # pi.set_epsilon(1/math.sqrt(1+0.25*i))

        start_time = time.time()
        policy_learner.find_policy()
        end_time = time.time()
        
        episode_duration = end_time - start_time
        episode_durations.append(episode_duration)

        value_function_drawer.update()
        greedy_optimal_policy_drawer.update()

        print(pi.epsilon())
        print(policy_learner._pi.epsilon())
        
        print(
            f"""

            ========================================================
            Episode: {i}
            Duration: {round(episode_duration,4)}
            Epsilon={pi.epsilon()}; alpha={policy_learner.alpha()}
            ========================================================

            """
        )
        
        pi.set_epsilon(1/math.sqrt(1+0.25*i))

    plt.plot(episode_durations, label="Episode Timings")

    plt.xlabel("Episode")
    plt.ylabel("Time (seconds)")
    
    plt.title("Time required for each episode")
    plt.legend()
    plt.show()
    
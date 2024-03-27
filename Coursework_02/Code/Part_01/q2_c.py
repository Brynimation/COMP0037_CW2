#!/usr/bin/env python3

'''
Created on 9 Mar 2023

@author: ucacsjj
'''

import math
import numpy as np
from p1.low_level_policy import LowLevelPolicy

from common.scenarios import corridor_scenario

from common.airport_map_drawer import AirportMapDrawer


from td.q_learner import QLearner

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer
from generalized_policy_iteration.tabular_value_function import TabularValueFunction
from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

def create_optimal_policy(env):
    optimal_policy = env.initial_policy()
    urs = [(1, 1), (18, 2), (11, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (20, 2)]
    ups = [(19, 1), (19, 2)]
    for x in range(0, 20):
        for y in range(0, 6):
            if (x, y) in urs:
                print(f'({x},{y})')
                optimal_policy.set_greedy_optimal_action(x, y, LowLevelActionType.MOVE_UP_RIGHT)
                optimal_policy.set_action(x, y, LowLevelActionType.MOVE_UP_RIGHT)
            elif (x, y) in ups:
                print(f'({x},{y})')
                optimal_policy.set_greedy_optimal_action(x, y, LowLevelActionType.MOVE_UP)
                optimal_policy.set_action(x, y, LowLevelActionType.MOVE_UP)
            else:
                optimal_policy.set_greedy_optimal_action(x, y, LowLevelActionType.MOVE_RIGHT)
                optimal_policy.set_action(x, y, LowLevelActionType.MOVE_RIGHT)
    return optimal_policy
def CompareValueFunctionMSE(v : TabularValueFunction, vTruth :TabularValueFunction) -> float:
    v_values = np.array([[v.value(x, y) for y in range(v._height)] for x in range(v._width)])
    v_truth_values = np.array([[vTruth.value(x, y) for y in range(v._height)] for x in range(v._width)])

    # Replace NaN values with zero in both arrays
    v_values_no_nan = np.nan_to_num(v_values, nan=0.0)
    v_truth_values_no_nan = np.nan_to_num(v_truth_values, nan=0.0)

    # Compute MSE with NaNs set to zero
    mse = np.mean((v_values_no_nan - v_truth_values_no_nan) ** 2)
    return mse
if __name__ == '__main__':
    filePlot = "2cdata.txt"
    airport_map, drawer_height = corridor_scenario()

    # Show the scenario        
    airport_map_drawer = AirportMapDrawer(airport_map, drawer_height)
    airport_map_drawer.update()
    
    # Create the environment
    env = LowLevelEnvironment(airport_map)
    optimal_policy = create_optimal_policy(env)
    # Extract the initial policy. This is e-greedy

    policy_learner = QLearner(env)  
    policy_learner.reset() 

    policy_learner.set_initial_policy(optimal_policy)
    policy_learner.set_target_policy(optimal_policy)
    policy_learner.set_alpha(0.1)
    policy_learner.set_experience_replay_buffer_size(32)
    policy_learner.set_number_of_episodes(32)

    # The drawers for the state value and the policy
    value_function_drawer = ValueFunctionDrawer(policy_learner.value_function(), drawer_height)    
    greedy_optimal_policy_drawer = LowLevelPolicyDrawer(policy_learner.policy(), drawer_height)
    
    for i in range(40):
        optimal_policy.set_epsilon(0.00)
        policy_learner.find_policy(False)
        value_function_drawer.update()
        greedy_optimal_policy_drawer.update()
        #policy_learner.set_target_policy(optimal_policy)

        
    #policy_learner.set_target_policy(optimal_policy)
        #print(f"epsilon={1/(math.sqrt(i*0.25+1))};alpha={policy_learner.alpha()}")
    value_function_drawer.update()
    greedy_optimal_policy_drawer.update()
    value_function_drawer.save_screenshot(f"2c_value_func_eps_optimal_.pdf")
    greedy_optimal_policy_drawer.save_screenshot(f"2c_policy_eps_optimal_.pdf")

    optimal_v = policy_learner.value_function()
    #optimal_policy = LowLevelPolicy("optimal_policy", env, 0.05)
    
    # Select the controller
    # These values worked okay for me.
    episode_counts = [1, 2, 4, 8, 16, 32, 64, 128]
    outer_iterations = [5, 10, 20, 40, 80, 160, 500, 1000, 2000, 5000]

    for ep_count in episode_counts:
        for iterations in outer_iterations:
            print(f'ep count: {ep_count}, iterations: {iterations}')
            pi = env.initial_policy()
            
    
            # Select the controller
            policy_learner = QLearner(env)  
            policy_learner.reset() 
            policy_learner.set_initial_policy(pi)


            policy_learner.set_alpha(0.1)
            policy_learner.set_experience_replay_buffer_size(32)
            policy_learner.set_number_of_episodes(ep_count)

            # The drawers for the state value and the policy
            value_function_drawer = ValueFunctionDrawer(policy_learner.value_function(), drawer_height)    
            greedy_optimal_policy_drawer = LowLevelPolicyDrawer(policy_learner.policy(), drawer_height)
            
            for i in range(iterations):
                policy_learner.find_policy()
                #value_function_drawer.update()
                #greedy_optimal_policy_drawer.update()
                pi.set_epsilon(1/math.sqrt(1+0.25*i))
                #print(f"epsilon={1/math.sqrt(1+i)};alpha={policy_learner.alpha()}")
                #policy_learner.set_target_policy(optimal_policy)   
            current_v = policy_learner._v
                
            #policy_learner.set_target_policy(optimal_policy)
                #print(f"epsilon={1/(math.sqrt(i*0.25+1))};alpha={policy_learner.alpha()}")
            value_function_drawer.update()
            greedy_optimal_policy_drawer.update()
            value_function_drawer.save_screenshot(f"2c_value_func_eps_{ep_count}_outerLoops_{iterations}_.pdf")
            greedy_optimal_policy_drawer.save_screenshot(f"2c_policy_eps_{ep_count}_outerLoops_{iterations}_.pdf")
            MSEerror = CompareValueFunctionMSE(optimal_v, current_v)
            with open(filePlot, 'a') as fp:
                fp.write(f'numEps:{ep_count}, numIterations:{iterations}, MSE={MSEerror}')
            

        
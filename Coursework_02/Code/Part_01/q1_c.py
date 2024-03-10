#!/usr/bin/env python3

'''
Created on 7 Mar 2023

@author: steam
'''

from common.scenarios import test_three_row_scenario
from common.airport_map_drawer import AirportMapDrawer

from td.td_policy_predictor import TDPolicyPredictor
from monte_carlo.on_policy_mc_predictor import OnPolicyMCPredictor
from monte_carlo.off_policy_mc_predictor import OffPolicyMCPredictor

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer
from generalized_policy_iteration.policy_evaluator import PolicyEvaluator

import numpy as np
from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer
from generalized_policy_iteration.tabular_value_function import TabularValueFunction

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
    filePlot = "epsilondata.txt"
    airport_map, drawer_height = test_three_row_scenario()
    env = LowLevelEnvironment(airport_map)
    env.set_nominal_direction_probability(0.8)

    # Policy to evaluate
    pi = env.initial_policy()
    pi.set_epsilon(0)
    pi.set_action(14, 1, LowLevelActionType.MOVE_DOWN)
    pi.set_action(14, 2, LowLevelActionType.MOVE_DOWN)  
    
    # Policy evaluation algorithm
    pe = PolicyEvaluator(env)
    pe.set_policy(pi)
    v_pe = ValueFunctionDrawer(pe.value_function(), drawer_height)  
    pe.evaluate()
    v_pe.update()
    # Calling update a second time clears the "just changed" flag
    # which means all the digits will be rendered in black
    v_pe.update()  
    
    # Off policy MC predictors
    
    epsilon_b_values = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 0.99, 1.0]
    num_episodes = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 10000]
    num_values = len(epsilon_b_values)
    for numEps in num_episodes:
        with open(filePlot, 'a') as fp:
            fp.write(f'{numEps}\n')
        mc_predictors = [None] * num_values
        mse_errors = [None] * num_values
        mc_drawers = [None] * num_values

        for i in range(num_values):
            mc_predictors[i] = OffPolicyMCPredictor(env)
            mc_predictors[i].set_use_first_visit(True)
            mc_predictors[i].set_number_of_episodes(numEps)
            b = env.initial_policy()
            b.set_epsilon(epsilon_b_values[i])
            mc_predictors[i].set_target_policy(pi)
            mc_predictors[i].set_behaviour_policy(b)
            mc_predictors[i].set_experience_replay_buffer_size(64)
            print(mc_predictors[i]._number_of_episodes)
            mc_drawers[i] = ValueFunctionDrawer(mc_predictors[i].value_function(), drawer_height)
            
        for e in range(1):
            for i in range(num_values):
                mc_predictors[i].evaluate()
                mc_drawers[i].update()
        for i in range(num_values):
            mse_errors[i] = CompareValueFunctionMSE(mc_predictors[i].value_function(), pe.value_function())
        with open(filePlot, 'a') as fp:
            fp.write(f'{mse_errors}\n')
        for i in range(num_values):
            mc_drawers[i].save_screenshot(f"mc-off-{int(epsilon_b_values[i]*10):03}-pe-{numEps}.pdf")
    v_pe.save_screenshot("q1_c_truth_pe.pdf")
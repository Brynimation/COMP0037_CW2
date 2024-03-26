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

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
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
    v_pe.update()  

    # Range of alpha values    
    alpha_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    
    num_values = len(alpha_values)
    
    td_predictors = [None] * num_values
    td_drawers = [None] * num_values
    
    # TD policy predictor
    for i in range (num_values):
        td_predictors[i] = TDPolicyPredictor(env)
        td_predictors[i].set_experience_replay_buffer_size(64)
        td_predictors[i].set_alpha(alpha_values[i])
        td_predictors[i].set_target_policy(pi)
        td_drawers[i] = ValueFunctionDrawer(td_predictors[i].value_function(), drawer_height)


    def get_current_value_functions(dpb: PolicyEvaluator):

        environment_map = dpb._environment.map()
        value_functions = np.zeros((environment_map.width(), environment_map.height()))

        for x in range(environment_map.width()):            
            for y in range(environment_map.height()):

                value_functions[x, y] = dpb.value_function().value(x, y)

        np.nan_to_num(value_functions, 0.0)

        return value_functions


    pe_value_functions = get_current_value_functions(pe)


    def get_rms_error(estimated_values):
            
            return np.sqrt(np.mean((np.array(estimated_values) - np.array(pe_value_functions))**2))


    alpha_errors = {alpha: [] for alpha in alpha_values}

    for e in range(400):
        
        for i, alpha in enumerate(alpha_values):
            
            td_predictors[i].evaluate()
            
            current_value_functions_estimate = get_current_value_functions(td_predictors[i])
            
            error = get_rms_error(current_value_functions_estimate)
            
            alpha_errors[alpha].append(error)

            td_drawers[i].update()        
 
    v_pe.save_screenshot("truth_pe.pdf")
    
    for i in range(num_values):
        td_drawers[i].update()
        td_drawers[i].save_screenshot(f"td-{alpha_values[i]}-pe.pdf")
    
    for alpha in alpha_values:
        plt.plot(alpha_errors[alpha], label=f"alpha = {alpha}")

    plt.xlabel("Episodes")
    plt.ylabel("Empirical RMS error (averaged over states)")
    plt.title("Empirical RMS error / Episodes")

    plt.legend()
    plt.show()

#!/usr/bin/env python3

'''
Created on 7 Mar 2023

@author: steam
'''

from common.scenarios import test_three_row_scenario
from common.airport_map_drawer import AirportMapDrawer

from monte_carlo.on_policy_mc_predictor import OnPolicyMCPredictor
from monte_carlo.off_policy_mc_predictor import OffPolicyMCPredictor

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer
from generalized_policy_iteration.policy_evaluator import PolicyEvaluator

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

from generalized_policy_iteration.tabular_value_function import TabularValueFunction
import math
import numpy as np

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
    pe.set_max_policy_evaluation_steps_per_iteration(100000)
    pe.evaluate()
    v_pe.update()  
    v_pe.update()  
    
    
    file = "data.txt"
    filePlot = "plottingdata.txt"
    evals = [1]
    num_episodes = [1000, 10000, 100000, 1000000]
    for e in evals:
        with open(filePlot, 'a') as fp:
            fp.write(f'{e}\n')
        for i in range(2):
            firstVisit = i == 0
            with open(filePlot, 'a') as fp:
                fp.write(f'first_visit = {firstVisit}\n')
            mcppErrors = []
            mcopErrors = []

            for j in range(len(num_episodes)):
                    # On policy MC predictor
                mcpp = OnPolicyMCPredictor(env)
                mcpp.set_target_policy(pi)
                mcpp.set_experience_replay_buffer_size(64)
                
                

                
                # Off policy MC predictor
                mcop = OffPolicyMCPredictor(env)
                mcop.set_target_policy(pi)
                mcop.set_experience_replay_buffer_size(64)
                b = env.initial_policy()
                b.set_epsilon(0.2)
                mcop.set_behaviour_policy(b)

                # Q1b: Experiment with this value
                mcop.set_use_first_visit(firstVisit)
                mcop.set_number_of_episodes(num_episodes[j])
                # Q1b: Experiment with this value
                mcpp.set_use_first_visit(firstVisit)
                mcpp.set_number_of_episodes(num_episodes[j])
                v_mcpp = ValueFunctionDrawer(mcpp.value_function(), drawer_height)
                v_mcop = ValueFunctionDrawer(mcop.value_function(), drawer_height)
                    
                for k in range(e):
                    mcpp.evaluate()
                    v_mcpp.update()
                    mcop.evaluate()
                    v_mcop.update()
                
                off_policy_error = CompareValueFunctionMSE(mcop.value_function(), pe.value_function())
                on_policy_error = CompareValueFunctionMSE(mcpp.value_function(), pe.value_function())

                mcopErrors.append(off_policy_error)
                mcppErrors.append(on_policy_error)
                text = f'{e}, {firstVisit}, {num_episodes[j]}: off policy error: {off_policy_error}, on policy error: {on_policy_error}\n'
                with open(file, 'a') as f:
                    f.write(text)
                # Sample way to generate outputs    
                v_mcop.save_screenshot(f'q1_b_mc-off_pe._{firstVisit}_{num_episodes[j]}_e={e}.pdf')
                v_mcpp.save_screenshot(f'q1_b_mc-on_pe_{firstVisit}_{num_episodes[j]}_e={e}.pdf')
            with open(filePlot, 'a') as fp:
                fp.write(f'on policy: {mcppErrors}\n')
                fp.write(f'off policy: {mcopErrors}\n')
                fp.close()
    v_pe.save_screenshot("q1_b_truth_pe.pdf")
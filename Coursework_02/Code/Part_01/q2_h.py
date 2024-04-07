#!/usr/bin/env python3

'''
Created on 9 Mar 2023

@author: ucacsjj
'''

import os, math

from common.scenarios import corridor_scenario
from common.airport_map_drawer import AirportMapDrawer

from q2_c import create_optimal_policy, CompareValueFunctionMSE

from td.sarsa import SARSA
from td.q_learner import QLearner

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer
import matplotlib.pyplot as plt

if __name__ == '__main__':
    filePlot = "2h_data.txt"

    if os.path.exists(filePlot):
        with open(filePlot, 'r') as f:
            data = f.readlines()
            data = [float(row.split("=")[1].replace("\n", "")) for row in data]
            print(data)

        plt.plot(data, label="MSE")

        plt.xlabel("Iteration")
        plt.ylabel("MSE")
        
        plt.title("MSE of SARSA by iteration number")
        plt.legend()
        plt.savefig("./plot.png")

    airport_map, drawer_height = corridor_scenario()

    # Show the scenario        
    airport_map_drawer = AirportMapDrawer(airport_map, drawer_height)
    airport_map_drawer.update()
    
    # Create the environment
    env = LowLevelEnvironment(airport_map)
    optimal_policy = create_optimal_policy(env)

    optimal_policy_learner = QLearner(env)  
    optimal_policy_learner.reset() 

    optimal_policy_learner.set_initial_policy(optimal_policy)
    optimal_policy_learner.set_target_policy(optimal_policy)
    optimal_policy_learner.set_alpha(0.1)
    optimal_policy_learner.set_experience_replay_buffer_size(32)
    optimal_policy_learner.set_number_of_episodes(32)
    
    for i in range(40):
        optimal_policy.set_epsilon(0.00)
        optimal_policy_learner.find_policy(False)
    
    optimal_v = optimal_policy_learner.value_function()

    # Specify array of learners, renderers and policies
    learners = [None] * 2
    v_renderers = [None] * 2
    p_renderers = [None] * 2 
    pi = [None] * 2
    
    pi[0] = env.initial_policy()
    pi[0].set_epsilon(1)
    learners[0] = SARSA(env)
    learners[0].set_alpha(0.5)
    learners[0].set_experience_replay_buffer_size(64)
    learners[0].set_number_of_episodes(32)
    learners[0].set_initial_policy(pi[0])
    v_renderers[0] = ValueFunctionDrawer(learners[0].value_function(), drawer_height)    
    p_renderers[0] = LowLevelPolicyDrawer(learners[0].policy(), drawer_height)
    
    
    pi[1] = env.initial_policy()
    pi[1].set_epsilon(1)
    learners[1] = QLearner(env)
    learners[1].set_alpha(0.5)
    learners[1].set_experience_replay_buffer_size(64)
    learners[1].set_number_of_episodes(32)
    learners[1].set_initial_policy(pi[1])      
    v_renderers[1] = ValueFunctionDrawer(learners[1].value_function(), drawer_height)    
    p_renderers[1] = LowLevelPolicyDrawer(learners[1].policy(), drawer_height)

    for i in range(10000):
        print(i)
        for l in range(2):
            learners[l].find_policy()
            v_renderers[l].update()
            p_renderers[l].update()
            pi[l].set_epsilon(1/math.sqrt(1+0.25*i))

            current_v = learners[l]._v


            model_name = "sarsa" if l == 0 else "qlearner"
            v_renderers[l].save_screenshot(f"value_{model_name}_iteration_{str(i)}.pdf")
            p_renderers[l].save_screenshot(f"policy_{model_name}_iteration_{str(i)}_.pdf")
            if model_name == "sarsa":
                MSEerror = CompareValueFunctionMSE(optimal_v, current_v)
                with open(filePlot, 'a') as fp:
                    fp.write(f'MSE={MSEerror}\n')
        
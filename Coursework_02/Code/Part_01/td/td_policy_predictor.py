'''
Created on 8 Mar 2023

@author: ucacsjj
'''

from monte_carlo.episode_sampler import EpisodeSampler

from .td_algorithm_base import TDAlgorithmBase

class TDPolicyPredictor(TDAlgorithmBase):

    def __init__(self, environment):
        
        TDAlgorithmBase.__init__(self, environment)
        
        self._minibatch_buffer= [None]
                
    def set_target_policy(self, policy):        
        self._pi = policy        
        self.initialize()
        self._v.set_name("TDPolicyPredictor")
        
    def evaluate(self):
        
        episode_sampler = EpisodeSampler(self._environment)
        
        for episode in range(self._number_of_episodes):

            # Choose the start for the episode            
            start_x, start_a  = self._select_episode_start()
            self._environment.reset(start_x) 
            
            # Now sample it
            new_episode = episode_sampler.sample_episode(self._pi, start_x, start_a)

            # If we didn't terminate, skip this episode
            if new_episode.terminated_successfully() is False:
                continue
            
            # Update with the current episode
            self._update_value_function_from_episode(new_episode)
            
            # Pick several randomly from the experience replay buffer and update with those as well
            for _ in range(min(self._replays_per_update, self._stored_experiences)):
                episode = self._draw_random_episode_from_experience_replay_buffer()
                self._update_value_function_from_episode(episode)
                
            self._add_episode_to_experience_replay_buffer(new_episode)
            
    def _update_value_function_from_episode(self, episode):

        # Q1e:
        # Complete implementation of this method
        # Each time you update the state value function, you will need to make a
        # call of the form:
        #
        # self._v.set_value(x_cell_coord, y_cell_coord, new_v)

        # Example to show how to extract coordinates; this does not do anything useful
        # coords = episode.state(0).coords()
        # new_v = 0
        # self._v.set_value(coords[0], coords[1], new_v)

        for step in range(episode.number_of_steps() - 1):
            
            state = episode.state(step)
            state_coords = state.coords()

            next_state = episode.state(step + 1)
            next_state_coords  = next_state.coords()

            v_next_state = self._v.value(next_state_coords[0], next_state_coords[1]) 
            
            reward = episode.reward(step)

            # old V(S)
            old_v = self._v.value(state_coords[0], state_coords[1])
            
            # V(S) <- V(S) + alpha * [R + gamma * V(S') - V(S)]
            new_v = old_v + self.alpha() * (reward + self.gamma() * v_next_state - old_v)
            
            self._v.set_value(state_coords[0], state_coords[1], new_v)


"""
Adapter from our custom game interface to the standard tensorflow environment interface.

Run this script to start it with a random-agent. Just to check if the adapter is working
"""

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import utils

import game


class RaceGameEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._intern_env = game.GameDing()
        self._target_type = np.float32
        #observation_shape = (self._intern_env.max_diff_x*2, self._intern_env.max_diff_y*2)
        observation_shape = (41,2)
        observation_min = (-self._intern_env.WINDOWWIDTH, -self._intern_env.WINDOWHEIGHT)
        observation_max = (self._intern_env.WINDOWWIDTH, self._intern_env.WINDOWHEIGHT)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=8, name='action') # positive, neutral, negative action
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=observation_shape, dtype=self._target_type, minimum=observation_min, maximum=observation_max, name='observation')
        self._state = 0.0
        self._episode_ended = True

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _convert_to_output_format(self, inp):
        return np.array(inp, dtype=self._target_type)
    
    def _fetch_observation(self):
        last_observation = self._intern_env.observation["position_vectors"]
        own_pos = np.array(self._intern_env.observation["own_pos"], ndmin=2)
        final = np.concatenate((own_pos, last_observation))
        return self._convert_to_output_format(final)

    def _action_converter(self, network_action: np.int8) -> (int, int):
        "Cast from the network space (9) to the game space (3x3)"
        actions = 3
        a = network_action%actions
        b = int(network_action/3)
        return (a, b)
        #return (network_action, 1)

    def _reset(self):
        self._state = 0.0
        self._episode_ended = False
        self._intern_env.reset()
        return ts.restart(self._fetch_observation())

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        a, b = self._action_converter(action)
        left_right = game.LeftRight(a - 1)
        #left_right = game.LeftRight(action - 1)
        up_down = game.FrontBack(b - 1)
        #up_down = game.FrontBack.NEUTRAL
        self._episode_ended = self._intern_env.next_round(left_right, up_down)
        if not self._episode_ended:
            self._state += 1.0

        if self._episode_ended:
            return ts.termination(self._fetch_observation(), reward=-25.0)
        else:
            return ts.transition(self._fetch_observation(), reward=0.25)

if __name__ == "__main__":
    print("Try to validate env")
    environment = RaceGameEnv()
    utils.validate_py_environment(environment, episodes=5)
    print("Environment validated")

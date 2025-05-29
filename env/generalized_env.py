import random
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
import gymnasium as gym
import numpy as np

# This class is a wrapper for the Overcooked environment that allows for
# generalized training across multiple layouts. It randomly selects a layout
# from a list of layouts each time the environment is reset. This is useful
# for training agents in a more diverse set of scenarios, which can help
# improve their generalization capabilities.

class GeneralizedOvercooked:
    def __init__(self, layouts, info_level=0, horizon=400):
        self.envs = []
        self.layout_names = layouts
        for layout in layouts:
            base_mdp = OvercookedGridworld.from_layout_name(layout)
            base_env = OvercookedEnv.from_mdp(base_mdp, info_level=info_level, horizon=horizon)
            env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
            self.envs.append(env)
        self.cur_env = self.envs[0]
        self.current_layout_name = layouts[0]  # Aggiunto

        self.observation_space, self.action_space = self.cur_env.observation_space, self.cur_env.action_space

    def reset(self):
        idx = random.randint(0, len(self.envs)-1)
        self.cur_env = self.envs[idx]
        self.current_layout_name = self.layout_names[idx]  # Aggiornato
        return self.cur_env.reset()

    def step(self, *args):
        return self.cur_env.step(*args)

    def render(self, *args):
        return self.cur_env.render(*args)

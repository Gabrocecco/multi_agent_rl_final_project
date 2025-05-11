import random
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
import gym
import numpy as np

class GeneralizedOvercooked(gym.Env):
    def __init__(self, layouts, info_level=0, horizon=400):
        self.envs = []
        for layout in layouts:
            mdp = OvercookedGridworld.from_layout_name(layout)
            base_env = OvercookedEnv.from_mdp(mdp, info_level=info_level, horizon=horizon)
            env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
            self.envs.append(env)

        self.cur_env = None
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.envs[0].observation_space.shape[0] * 2,))
        self.action_space = gym.spaces.Discrete(self.envs[0].action_space.n)

        self.reset()

    def reset(self):
        self.cur_env = random.choice(self.envs)
        obs = self.cur_env.reset()
        return np.concatenate(obs)

    def step(self, action_joint):
        obs, reward, done, info = self.cur_env.step(action_joint)
        return np.concatenate(obs), reward, done, info

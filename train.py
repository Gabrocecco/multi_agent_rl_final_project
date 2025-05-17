from env.generalized_env import GeneralizedOvercooked
from models.ppo import PPOAgent
import numpy as np
import tensorflow as tf

env = GeneralizedOvercooked(['cramped_room', 'counter_circuit_o_1order', 'coordination_ring'])

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

agent = PPOAgent(obs_dim, act_dim)

n_episodes = 1000
max_steps = 400

for ep in range(n_episodes):
    obs = env.reset()
    obs_batch, act_batch, old_probs, rewards, dones = [], [], [], [], []

    for t in range(max_steps):
        a0, p0 = agent.select_action(obs[:obs_dim // 2])
        a1, p1 = agent.select_action(obs[obs_dim // 2:])
        actions = [a0, a1]
        next_obs, reward, done, _ = env.step(actions)

        obs_batch.append(obs)
        act_batch.append(a0)  # Assumiamo entrambi usino stessa rete
        old_probs.append(p0)
        rewards.append(reward)
        dones.append(done)

        obs = next_obs
        if done:
            break

    last_value = agent.value_net(tf.convert_to_tensor(obs, dtype=tf.float32)).numpy()
    returns = agent.compute_returns(rewards, dones, last_value)
    agent.update(obs_batch, act_batch, old_probs, returns)

    if ep % 10 == 0:
        print(f"Episode {ep}, mean reward: {np.mean(rewards):.2f}")

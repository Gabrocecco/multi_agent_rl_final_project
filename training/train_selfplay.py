from env.generalized_env import GeneralizedOvercooked
from agents.ppo_tf import PPOAgent
import numpy as np
import tensorflow as tf

print("\n\n\n\n\n\n\nTraining PPO agent on Generalized Overcooked environment. " \
"This is a self-play implementation where both agents share the same policy.")

# Initialize environment
# Note: The layouts are defined in the OvercookedAI library. You can find them in the
# overcooked_ai_py/overcooked_ai_py/layouts directory.
layouts = ['cramped_room', 'counter_circuit_o_1order', 'coordination_ring']

# Initialize the GeneralizedOvercooked environment with the specified layouts
env = GeneralizedOvercooked(layouts)
print(f"Using {len(env.envs)} layouts: {layouts}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Note: The observation space and action space are defined in the OvercookedAI library.
# The observation space is a dictionary with keys 'both_agent_obs' and 'both_agent_actions'.
# The 'both_agent_obs' key contains a list of observations for both agents, and the
# 'both_agent_actions' key contains a list of actions for both agents.
# The action space is a discrete space with the number of actions defined by the
# OvercookedAI library.
# Note: The action space is a discrete space with the number of actions defined by the
# OvercookedAI library.
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
print(f"Observation dimension: {obs_dim}")
print(f"Action dimension: {act_dim}")

# Initialize PPO agent (shared between both players)
agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim)

n_episodes = 100
max_steps = 10
# n_episodes = 1
# max_steps = 1
print(f"Training for {n_episodes} episodes with max {max_steps} steps each.")


# Training loop
for ep in range(n_episodes):
    obs = env.reset()
    # print(f"Initial observation: {obs}")
    # print(f"Observation shape: {obs['both_agent_obs'][0].shape}")
    # print(f"Observation keys: {obs.keys()}")

    obs_batch, act_batch, old_probs, rewards, dones = [], [], [], [], []
    episode_reward = 0

    print(f"Episode {ep+1}/{n_episodes}")

    for t in range(max_steps):
        # print(f"[DEBUG] Type of obs: {type(obs)}, obs shape: {getattr(obs, 'shape', 'no shape')}")
        # print(obs.keys())

        # print(f"Step {t+1}/{max_steps}")

        # Correctly extract agent-specific observations from dict
        a0, p0 = agent.select_action(obs['both_agent_obs'][0])
        a1, p1 = agent.select_action(obs['both_agent_obs'][1])
        # print(f"Agent 0 action: {a0}, prob: {p0}")
        # print(f"Agent 1 action: {a1}, prob: {p1}")
        
        actions = [a0, a1]

        next_obs, reward, done, _ = env.step(actions)
        # print(f"Next observation: {next_obs}")
        # print(f"Reward: {reward}, Done: {done}")

        # Append observations, actions, and rewards to the batch
        # Note: The observations are from the perspective of agent 0
        # and the actions are from the perspective of both agents
        # why?  
        # because we are training the agent to play against itself
        # and thus we only need the observations and actions from agent 0's perspective
        obs_batch.append(obs['both_agent_obs'][0])  # or [both_agent_obs[0], both_agent_obs[1]] if you prefer
        act_batch.append(a0)  # Usa solo l'azione del primo agente
        old_probs.append(p0)
        rewards.append(reward)
        dones.append(done)

        obs = next_obs
        episode_reward += reward
        # print(f"Episode reward: {episode_reward}")
        # print(f"Current reward: {reward}")
        # print(f"Done: {done}")
        if done:
            break

    # Bootstrapped value for last state
    # last_value = agent.value(tf.convert_to_tensor([obs], dtype=tf.float32)).numpy()[0, 0]

    # Compute bootstrap value for the last state from agent 0's perspective
    last_value = agent.value(tf.convert_to_tensor([obs['both_agent_obs'][0]], dtype=tf.float32)).numpy()[0, 0]
    # print(f"Last value: {last_value}")
    # Compute returns
    # Note: The returns are computed using the rewards and dones from agent 0's perspective
    # This is a simplification, as both agents share the same policy and thus the same returns
    # but in a more complex scenario, you might want to compute returns for both agents separately
    # and then average them or use some other method to combine them.
    returns = agent.compute_returns(rewards, dones, last_value)
    # print(f"Returns: {returns}")
    # PPO update
    # Note: The update function should be modified to accept the observations and actions
    # from both agents if you want to train them separately, but in this case, we are
    # assuming they share the same policy and thus the same observations and actions
    # agent.update(obs_batch, act_batch, old_probs, returns)
    # Update the agent with the observations and actions from agent 0
    agent.update(obs_batch, act_batch, old_probs, returns)
    # print(f"Updated agent with {len(obs_batch)} samples.")
    
    # Print mean reward for the episode
    if ep % 10 == 0:
        print(f"[Episode {ep}] Mean reward: {np.mean(rewards):.2f}  Total reward: {episode_reward:.2f}")

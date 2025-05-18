from env.generalized_env import GeneralizedOvercooked
from agents.ppo_tf import PPOAgent
import numpy as np
import tensorflow as tf
import os
import csv

print("\n\n\n\n\n\n\nTraining PPO agent on Generalized Overcooked environment. "
      "This is a self-play implementation where both agents share the same policy.")

# === Environment setup ===
layouts = ['cramped_room']
env = GeneralizedOvercooked(layouts)
print(f"Using {len(env.envs)} layouts: {layouts}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
print(f"Observation dimension: {obs_dim}")
print(f"Action dimension: {act_dim}")

# === Agent setup ===
agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, arch_variant='conv')

# === Check for resume ===
resume = True  # <-- Set to True to resume from checkpoint
resume_episode = 2000
if resume:
    checkpoint_path = f"checkpoints/policy_ep{resume_episode}.weights.h5"
    if os.path.exists(checkpoint_path):
        agent.policy.load_weights(checkpoint_path)
        agent.value.load_weights(f"checkpoints/value_ep{resume_episode}.weights.h5")
        print(f"\nResumed training from checkpoint at episode {resume_episode}.")
        start_episode = resume_episode
    else:
        print("\nCheckpoint not found. Starting from scratch.")
        start_episode = 0
else:
    start_episode = 0

# === Training config ===
n_episodes = 3000
max_steps = 400
print(f"Training for {n_episodes} episodes with max {max_steps} steps each.")

# === Logging setup ===
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "train_log.csv")

if not os.path.exists(log_path):
    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "mean_reward", "total_reward", "decay_factor",
            "shaped_reward_first_step", "sparse_reward_first_step", "evaluation_reward_greedy"
        ])

# === Evaluation function ===
def evaluate_policy(agent, env, greedy=True):
    obs = env.reset()
    total_reward = 0
    for _ in range(max_steps):
        obs0 = obs['both_agent_obs'][0]
        obs1 = obs['both_agent_obs'][1]
        if greedy:
            logits0 = agent.policy(tf.convert_to_tensor([obs0], dtype=tf.float32))
            logits1 = agent.policy(tf.convert_to_tensor([obs1], dtype=tf.float32))
            a0 = tf.argmax(logits0[0]).numpy()
            a1 = tf.argmax(logits1[0]).numpy()
        else:
            a0, _ = agent.select_action(obs0)
            a1, _ = agent.select_action(obs1)
        obs, reward, done, _ = env.step([a0, a1])
        total_reward += reward
        if done:
            break
    return total_reward

# === Training loop ===
for ep in range(start_episode, n_episodes):
    obs = env.reset()
    obs_batch, act_batch, old_probs, rewards, dones = [], [], [], [], []
    episode_reward = 0
    action_counts = {i: 0 for i in range(act_dim)}
    decay_factor = max(0.0, 1.0 - ep / 2000)
    first_shaped = None
    first_sparse = None

    for t in range(max_steps):
        a0, p0 = agent.select_action(obs['both_agent_obs'][0])
        a1, p1 = agent.select_action(obs['both_agent_obs'][1])
        actions = [a0, a1]
        action_counts[a0] += 1

        next_obs, reward_sparse, done, info = env.step(actions)

        shaped = info.get('shaped_r_by_agent', [0, 0])[0]
        scale = 5.0 if ep < 1000 else 2.5 if ep < 2000 else 1.0
        weighted_shaped = decay_factor * (shaped * scale)
        total_reward = reward_sparse + weighted_shaped

        if t == 0:
            first_shaped = shaped
            first_sparse = reward_sparse
            if ep % 10 == 0:
                print(f"[Episode {ep}] Decay: {decay_factor:.2f} | Sparse: {reward_sparse:.2f} | Shaped: {shaped:.2f} | Total: {total_reward:.2f}")

        obs_batch.append(obs['both_agent_obs'][0])
        act_batch.append(a0)
        old_probs.append(p0)
        rewards.append(total_reward)
        dones.append(done)

        obs = next_obs
        episode_reward += total_reward
        if done:
            break

    print(f"Episode {ep+1}/{n_episodes}")
    print("Action distribution:", action_counts)

    last_value = agent.value(tf.convert_to_tensor([obs['both_agent_obs'][0]], dtype=tf.float32)).numpy()[0, 0]
    returns = agent.compute_returns(rewards, dones, last_value)
    agent.update(obs_batch, act_batch, old_probs, returns)

    mean_reward = np.mean(rewards)
    print(f"[Episode {ep}] Mean reward: {mean_reward:.2f}  Total reward: {episode_reward:.2f}")

    evaluation_reward = 0.0
    if (ep + 1) % 100 == 0:
        evaluation_reward = evaluate_policy(agent, env, greedy=True)
        print(f"[Evaluation @ episode {ep+1}] Total reward (greedy policy): {evaluation_reward:.2f}")

    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            ep, mean_reward, episode_reward, decay_factor,
            first_shaped, first_sparse, evaluation_reward
        ])

    # === Save model every 500 episodes ===
    if (ep + 1) % 500 == 0:
        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        agent.policy.save_weights(os.path.join(save_dir, f"policy_ep{ep+1}.weights.h5"))
        agent.value.save_weights(os.path.join(save_dir, f"value_ep{ep+1}.weights.h5"))
        print(f"\nCheckpoint saved at episode {ep+1}.")

# === Save final model ===
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)
agent.policy.save_weights(os.path.join(save_dir, "policy_final.weights.h5"))
agent.value.save_weights(os.path.join(save_dir, "value_final.weights.h5"))
print("\nFinal model weights saved in 'checkpoints/'")

# === Final greedy rollout and save as GIF ===
print("\nSimulating final policy (greedy)...")
eval_reward = evaluate_policy(agent, env, greedy=True)
print(f"Final greedy rollout total reward: {eval_reward:.2f}")

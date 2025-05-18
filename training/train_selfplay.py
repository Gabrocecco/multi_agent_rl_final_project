from env.generalized_env import GeneralizedOvercooked
from agents.ppo_tf import PPOAgent
import numpy as np
import tensorflow as tf
import os
import csv

import imageio
from PIL import Image
import pygame
# === Evaluate policy function  ===
import imageio
import tensorflow as tf

def evaluate_policy(agent, env, greedy=True, save_gif=False, gif_path="final_rollout.gif", steps = 400):
    import imageio

    obs = env.reset()
    total_reward = 0
    frames = []
    # steps = 1000

    print(f"\n--- Starting evaluation rollout (greedy = {greedy}) ---\n")

    for t in range(steps):
        obs0 = obs['both_agent_obs'][0]
        obs1 = obs['both_agent_obs'][1]

        # Select actions
        if greedy:
            logits0 = agent.policy(tf.convert_to_tensor([obs0], dtype=tf.float32))
            logits1 = agent.policy(tf.convert_to_tensor([obs1], dtype=tf.float32))
            a0 = tf.argmax(logits0[0]).numpy()
            a1 = tf.argmax(logits1[0]).numpy()
        else:
            a0, _ = agent.select_action(obs0)
            a1, _ = agent.select_action(obs1)

        # Step
        obs, reward, done, info = env.step([a0, a1])
        total_reward += reward

        # Log reward components
        sparse = info.get("sparse_r_by_agent", [0, 0])
        shaped = info.get("shaped_r_by_agent", [0, 0])
        delivered = info.get("num_orders_delivered", None)
        events = info.get("event_info", {})

        if sum(sparse) > 0 or sum(shaped) > 0:
            print(f"[STEP {t}] Sparse: {sparse} | Shaped: {shaped} | Delivered: {delivered}")
            print(f"           Events: {events}")

        # Render frame
        if save_gif:
            frame = env.envs[0].render()
            frames.append(frame)

        if done:
            print("[Done] Environment signaled done.")
            break

    if save_gif and frames:
        imageio.mimsave(gif_path, frames, fps=5)
        print(f"\n✅ Saved final policy rollout as GIF: {gif_path}")

    print(f"\n--- Evaluation complete ---\nTotal reward: {total_reward:.2f}\n")
    return total_reward



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
network = "deep"
use_shaping = True
use_decay = True
agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, arch_variant=network,
                 use_reward_shaping=use_shaping, use_decay=use_decay)
print(f"Using the {network} network")

# === Check for resume ===
resume = True  # <-- Set to True to resume from checkpoint
resume_episode = 6000
if resume:
    # checkpoint_path = f"checkpoints/policy_ep{resume_episode}.weights.h5"
    checkpoint_path_policy = "/home/gabro/Desktop/AAS/final_project/overcooked_rl/checkpoints/policy_PPO_deep_shaping-decay-6000_new_rl_final.weights.h5"
    checkpoint_path_value = "/home/gabro/Desktop/AAS/final_project/overcooked_rl/checkpoints/value_PPO_deep_shaping-decay-6000_new_rl_final.weights.h5"

    if os.path.exists(checkpoint_path_policy):
        # Costruisci i modelli
        dummy_input = tf.random.uniform((1, obs_dim))
        agent.policy(dummy_input)
        agent.value(dummy_input)

        # Carica i pesi
        agent.policy.load_weights(checkpoint_path_policy)
        # agent.value.load_weights(f"checkpoints/value_ep{resume_episode}.weights.h5")
        agent.value.load_weights(checkpoint_path_value)

        print(f"\nResumed training from checkpoint at episode {resume_episode}.")
        start_episode = resume_episode
    else:
        print("\nCheckpoint not found. Starting from scratch.")
        start_episode = 0
else:
    print("\nStarting from scratch.")
    start_episode = 0


# === Training config ===
n_episodes = 6000
max_steps = 400
print(f"Training for {n_episodes} episodes with max {max_steps} steps each.")

# === Define a unique tag for this experiment ===
shaping_tag = "shaping" if use_shaping else "noshaping"
decay_tag = "decay" if use_decay else "nodecay"
tag = f"PPO_{network}_{shaping_tag}-{decay_tag}-{n_episodes}_new_rl"

# === Logging setup ===
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"train_log_{tag}.csv")
if not os.path.exists(log_path):
    print("Log file does not exist.... create one\n")
    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "mean_reward", "total_reward", "decay_factor",
            "shaped_reward_first_step", "sparse_reward_first_step", "evaluation_reward_greedy"
        ])


# === Training loop ===
for ep in range(start_episode, n_episodes):
    obs = env.reset()
    obs_batch, act_batch, old_probs, rewards, dones = [], [], [], [], []
    episode_reward = 0
    action_counts = {i: 0 for i in range(act_dim)}
    decay_factor = max(0.0, 1.0 - ep / n_episodes) if agent.use_decay else 1.0
    first_shaped = None
    first_sparse = None
    dones_sparse_rewards = []
    dones_shaped_rewards = []
    for t in range(max_steps):
        a0, p0 = agent.select_action(obs['both_agent_obs'][0])
        a1, p1 = agent.select_action(obs['both_agent_obs'][1])
        actions = [a0, a1]
        action_counts[a0] += 1

        next_obs, reward_sparse, done, info = env.step(actions)

        shaped = info.get('shaped_r_by_agent', [0, 0])[0]
        # scale = 5.0 if ep < 1000 else 2.5 if ep < 2000 else 1.0 # try 1
        scale = 10.0 if ep < 1000 else 5.0 if ep < 3000 else 2.0

        if agent.use_reward_shaping:
            weighted_shaped = decay_factor * (shaped * scale)
            total_reward = reward_sparse + weighted_shaped
        else:
            total_reward = reward_sparse

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

        if agent.use_reward_shaping:
            dones_sparse_rewards.append(reward_sparse)
            dones_shaped_rewards.append(shaped * scale)
        # print(f"Episode {ep+1}/{n_episodes}")

    # === Action distribution with readable labels ===
    action_labels = {
        0: "STAY",
        1: "UP",
        2: "DOWN",
        3: "LEFT",
        4: "RIGHT",
        5: "INTERACT"
    }

    print("Action distribution:")
    for action_id, count in action_counts.items():
        label = action_labels.get(action_id, f"UNKNOWN_{action_id}")
        print(f"  {label:8}: {count}")

    # === Compute returns and update agent ===
    last_value = agent.value(tf.convert_to_tensor([obs['both_agent_obs'][0]], dtype=tf.float32)).numpy()[0, 0]
    returns = agent.compute_returns(rewards, dones, last_value)
    agent.update(obs_batch, act_batch, old_probs, returns)

    # === Stats ===
    mean_reward = np.mean(rewards)
    total_sparse_reward = sum(dones_sparse_rewards) if agent.use_reward_shaping else sum(rewards)
    total_shaped_reward = sum(dones_shaped_rewards) if agent.use_reward_shaping else 0.0
    episode_reward = total_sparse_reward + total_shaped_reward

    print(f"[Episode {ep}] Mean reward: {mean_reward:.2f}  Total reward: {episode_reward:.2f}")
    if agent.use_reward_shaping:
        print(f"  -> Total sparse reward: {total_sparse_reward:.2f}")
        print(f"  -> Total shaped reward: {total_shaped_reward:.2f}")

    # === Evaluation ===
    evaluation_reward = 0.0
    if (ep + 1) % 100 == 0:
        evaluation_reward = evaluate_policy(agent, env, greedy=True)
        print(f"[Evaluation @ episode {ep+1}] Total reward (greedy policy): {evaluation_reward:.2f}")

    # === Logging ===
    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            ep, mean_reward, episode_reward, decay_factor,
            first_shaped, first_sparse, evaluation_reward
        ])

    # === Save model every 250 episodes ===
    if (ep + 1) % 250 == 0:
        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        agent.policy.save_weights(os.path.join(save_dir, f"policy_{tag}_ep{ep+1}.weights.h5"))
        agent.value.save_weights(os.path.join(save_dir, f"value_{tag}_ep{ep+1}.weights.h5"))
        print(f"\nCheckpoint saved at episode {ep+1}.")


# === Save final model ===
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)
agent.policy.save_weights(os.path.join(save_dir, f"policy_{tag}_final.weights.h5"))
agent.value.save_weights(os.path.join(save_dir, f"value_{tag}_final.weights.h5"))


print("\nFinal model weights saved in 'checkpoints/'")

# === Final greedy rollout and save as GIF ===
gif_path = f"final_rollout_{tag}.gif"
greedy_status = False
print(f"\nSimulating final policy (greedy = {greedy_status})...")
# eval_reward = evaluate_policy(agent, env, greedy=False, save_gif=False, gif_path=gif_path, n_steps = max_steps)
for i in range(1):
    eval_reward = evaluate_policy(
            agent,
            env,
            greedy=greedy_status,
            save_gif=True,
            gif_path="rollouts/final_run.gif",
            steps = max_steps
        )
    print(f"Final greedy = {greedy_status} rollout total reward: {eval_reward:.2f}")
    # if(eval_reward > 0):
    #     print("REWARDDDDDDD\n\n\n")
    #     break



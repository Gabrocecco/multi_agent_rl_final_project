import os
import csv
import numpy as np
import tensorflow as tf
import datetime
from env.generalized_env import GeneralizedOvercooked
from agents.ppo_tf import PPOAgent
from training.evaluate_selfplay import evaluate_policy

# Main training funcion, modify parameters as needed.
def train(
    experiment_name="multilayout_run_20000_demo_test",
    network='deep',
    use_shaping=True,
    use_decay=True,
    total_episodes=20000,
    max_steps=400,
    resume=False,
    resume_episode=17300,
    resume_tag = "PPO_deep_shaping-decay-20000multilayout_run_20000_final.weights.h5",
    rollout_greedy=True,
):
    # --- MULTI-LAYOUT SETUP ---
    train_layouts = ["cramped_room", "asymmetric_advantages", "forced_coordination"]  # Inser here the layouts you want to train on
    tag = f"PPO_{network}_{'shaping' if use_shaping else 'noshaping'}-{'decay' if use_decay else 'nodecay'}-{total_episodes}{experiment_name}"

    # --- DIRS ---
    log_dir = "multilayout_logs"
    checkpoint_dir = "multilayout_checkpoints"
    rollout_dir = "multilayout_rollouts"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(rollout_dir, exist_ok=True)

    # --- FILES ---
    date_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_log_{tag}_{date_tag}.csv")
    gif_path = os.path.join(rollout_dir, f"final_rollout_{tag}.gif")

    # TensorBoard logging
    tb_log_dir = os.path.join("tensorboard_logs", f"{tag}_{date_tag}")
    summary_writer = tf.summary.create_file_writer(tb_log_dir)
    print(f"[TensorBoard] Logging at {tb_log_dir}")
    print(f"[LOG] Writing CSV log to: {log_path}")

    # --- ENVIRONMENT SETUP ---
    print("\nInitializing Overcooked environment with layouts:", train_layouts)
    env = GeneralizedOvercooked(train_layouts)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    print("\nTraining PPO agent on Overcooked")
    print(f"Layouts: {train_layouts}, Obs dim: {obs_dim}, Action dim: {act_dim}")

    # --- AGENT SETUP ---
    agent = PPOAgent(
        obs_dim, act_dim,
        arch_variant=network,
        use_reward_shaping=use_shaping,
        use_decay=use_decay,
        clip_ratio=0.15,
        lr=5e-4
    )
    start_episode = 0
    # Load the agent's policy and value networks from the specified checkpoint if resume = True
    if resume:
        print(f"\nResuming training from episode {resume_episode} with tag: {resume_tag}")
        try:
            # Ensure the agent is initialized before loading weights
            dummy_input = tf.random.uniform((1, obs_dim))
            agent.policy(dummy_input)
            agent.value(dummy_input)
            # Load the weights from the specified checkpoint
            agent.policy.load_weights(os.path.join(checkpoint_dir, f"policy_{resume_tag}"))
            agent.value.load_weights(os.path.join(checkpoint_dir, f"value_{resume_tag}"))
            start_episode = resume_episode
            print(f"Resumed from checkpoint at episode {resume_episode}")
        except Exception as e:
            print("Failed to load checkpoint:", e)

    # Logging header
    csv_header = [
        "episode", "layout", "mean_reward", "total_reward", "decay_factor",
        "evaluation_reward_greedy_mean", "evaluation_reward_greedy_std", "moving_avg_train_reward",
        "total_shaped_reward", "total_sparse_reward",
        "moving_avg_shaped_reward", "moving_avg_sparse_reward"
    ]
    # Create the CSV file and write the header
    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    # --- TRAINING LOOP ---
    print("\nStarting training loop...")
    rolling_eval_rewards = []
    best_eval = -float('inf')
    shaped_history = []
    sparse_history = []
    reward_history = []
    layout_history = []

    # Dictionaries to store rewards for each layout
    layout_reward_history = {layout: [] for layout in train_layouts}
    layout_shaped_history = {layout: [] for layout in train_layouts}
    layout_sparse_history = {layout: [] for layout in train_layouts}

    # Start training episodes
    print(f"\nTotal episodes: {total_episodes}, Max steps per episode: {max_steps}")
    for ep in range(start_episode, total_episodes):
        obs = env.reset()   # Reset the environment for a new episode
        # Extract the current layout name from the environment
        if hasattr(env, "current_layout_name"):
            current_layout = env.current_layout_name
        elif hasattr(env, "envs") and hasattr(env.envs[0], "layout_name"):
            current_layout = env.envs[0].layout_name
        else:
            current_layout = "unknown"
        layout_history.append(current_layout)

        # Initialize lists to store episode data
        obs_batch, act_batch, old_probs, rewards, dones = [], [], [], [], []

        # Decay factor for reward shaping
        # decay_factor = max(0.0, 1.0 - ep / 12000) if use_decay else 1.0
        decay_total = 18000   # oppure 16000 o anche 20000 se vuoi una piccola coda
        decay_factor = max(0.0, 1.0 - ep / decay_total) if use_decay else 1.0

        # Initialize lists for shaped and sparse rewards
        shaped_rewards, sparse_rewards = [], []
        episode_reward = 0

        # Run the episode for a maximum number of steps
        for t in range(max_steps):
            # Extract the observations for both agents
            a0, p0 = agent.select_action(obs['both_agent_obs'][0])  # Select action for agent 0
            a1, _ = agent.select_action(obs['both_agent_obs'][1])   # Select action for agent 1
            actions = [a0, a1]  # Combine actions for both agents
            # Take a step in the environment with the selected actions
            next_obs, reward_sparse, done, info = env.step(actions) 
            shaped = info.get('shaped_r_by_agent', [0, 0])[0] # shaped reward is shared by both agents, we take the first one

            # if ep < 2500:
            #     scale = 10.0
            # elif ep < 6000:
            #     scale = 5.0
            # else:
            #     scale = 2.0

            # Dynamic scaling based on episode number
            if ep < 4000:
                scale = 10.0
            elif ep < 10000:
                scale = 5.0
            elif ep < 16000:
                scale = 2.0
            else:
                scale = 1.0

            # Calculate the final reward with shaping and decay factor
            reward = reward_sparse + decay_factor * (shaped * scale) if use_shaping else reward_sparse

            # Append data to the episode lists
            obs_batch.append(obs['both_agent_obs'][0])
            act_batch.append(a0)
            old_probs.append(p0)
            rewards.append(reward)
            dones.append(done)

            # Append shaped and sparse rewards
            shaped_rewards.append(shaped * scale)
            sparse_rewards.append(reward_sparse)

            # Update the observation for the next step
            obs = next_obs
            episode_reward += reward

            if done:
                break

        # Calculate total rewards for the episode
        total_shaped_reward = sum(shaped_rewards)
        total_sparse_reward = sum(sparse_rewards)
        shaped_history.append(total_shaped_reward)
        sparse_history.append(total_sparse_reward)
        reward_history.append(episode_reward)

        # Append to layout-specific histories
        layout_reward_history[current_layout].append(episode_reward)
        layout_shaped_history[current_layout].append(total_shaped_reward)
        layout_sparse_history[current_layout].append(total_sparse_reward)

        # Calculate moving averages for the last 100 episodes 
        moving_avg_reward = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)
        moving_avg_shaped = np.mean(shaped_history[-100:]) if len(shaped_history) >= 100 else np.mean(shaped_history)
        moving_avg_sparse = np.mean(sparse_history[-100:]) if len(sparse_history) >= 100 else np.mean(sparse_history)

        # Calculate moving averages for each layout
        layout_moving_avg_reward = {layout: np.mean(rews[-100:]) if len(rews) >= 100 else np.mean(rews)
                                    for layout, rews in layout_reward_history.items()}
        layout_moving_avg_shaped = {layout: np.mean(rews[-100:]) if len(rews) >= 100 else np.mean(rews)
                                    for layout, rews in layout_shaped_history.items()}
        layout_moving_avg_sparse = {layout: np.mean(rews[-100:]) if len(rews) >= 100 else np.mean(rews)
                                    for layout, rews in layout_sparse_history.items()}

        print(f"Episode {ep+1}/{total_episodes} | Layout: {current_layout} | Sparse: {total_sparse_reward} | Shaped: {total_shaped_reward} | Total: {episode_reward:.2f}")

        last_value = agent.value(tf.convert_to_tensor([obs['both_agent_obs'][0]], dtype=tf.float32)).numpy()[0, 0]
        returns = agent.compute_returns(rewards, dones, last_value) # Compute returns for the episode
        agent.update(obs_batch, act_batch, old_probs, returns) # Update the agent with the collected data

        # === Greedy eval: average of 10 runs and std ===
        eval_reward_mean = 0.0
        eval_reward_std = 0.0
        if (ep + 1) % 100 == 0:
            eval_rewards = []
            for _ in range(10):
                reward = evaluate_policy(agent, env, greedy=True)
                eval_rewards.append(reward)
            eval_reward_mean = np.mean(eval_rewards)
            eval_reward_std = np.std(eval_rewards)
            print(f"[Eval] Mean over 10 greedy runs: {eval_reward_mean:.2f} (std: {eval_reward_std:.2f})")
            rolling_eval_rewards.append(eval_reward_mean)
            if len(rolling_eval_rewards) > 10:
                rolling_eval_rewards.pop(0)
            mean_eval = np.mean(rolling_eval_rewards)

            # Stampa anche le moving avg reward per layout
            print("Moving average reward per layout (last 100):")
            for layout in train_layouts:
                print(f"  {layout}: {layout_moving_avg_reward[layout]:.2f}")

            # # Early stopping
            # if mean_eval >= 50:
            #     print(f"\nEarly stopping at episode {ep+1}, mean greedy eval (last 10 evals): {mean_eval:.2f}")
            #     agent.policy.save_weights(os.path.join(checkpoint_dir, f"policy_{tag}_final.weights.h5"))
            #     agent.value.save_weights(os.path.join(checkpoint_dir, f"value_{tag}_final.weights.h5"))
            #     evaluate_policy(agent, env, greedy=rollout_greedy, max_steps=max_steps, save_gif=True, gif_path=gif_path)
            #     print("Training stopped early (target reached!)")
            #     if eval_reward_mean > best_eval:
            #         best_eval = eval_reward_mean
            #         agent.policy.save_weights(os.path.join(checkpoint_dir, f"policy_{tag}_best.weights.h5"))
            #         agent.value.save_weights(os.path.join(checkpoint_dir, f"value_{tag}_best.weights.h5"))
            #         print(f"Saved best model at episode {ep+1} (eval_reward={best_eval:.2f})")
            #     return

        # --- Logging CSV ---
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep, current_layout, np.mean(rewards), episode_reward, decay_factor,
                eval_reward_mean, eval_reward_std, moving_avg_reward,
                total_shaped_reward, total_sparse_reward,
                moving_avg_shaped, moving_avg_sparse
            ])

        # --- Logging TensorBoard ---
        with summary_writer.as_default():
            tf.summary.scalar('Train/TotalReward', episode_reward, step=ep)
            tf.summary.scalar('Train/MeanReward', np.mean(rewards), step=ep)
            tf.summary.scalar('Train/DecayFactor', decay_factor, step=ep)
            tf.summary.scalar('Train/MovingAvgReward', moving_avg_reward, step=ep)
            tf.summary.scalar('Train/TotalShapedReward', total_shaped_reward, step=ep)
            tf.summary.scalar('Train/TotalSparseReward', total_sparse_reward, step=ep)
            tf.summary.scalar('Train/MovingAvgShapedReward', moving_avg_shaped, step=ep)
            tf.summary.scalar('Train/MovingAvgSparseReward', moving_avg_sparse, step=ep)
            tf.summary.scalar('Eval/GreedyEvalReward', eval_reward_mean, step=ep)
            tf.summary.scalar('Eval/GreedyEvalStd', eval_reward_std, step=ep)
            tf.summary.text('Layout', str(current_layout), step=ep)
            # Log moving avg per layout
            for layout in train_layouts:
                tf.summary.scalar(f'Train/MovingAvgReward_{layout}', layout_moving_avg_reward[layout], step=ep)
                tf.summary.scalar(f'Train/MovingAvgShaped_{layout}', layout_moving_avg_shaped[layout], step=ep)
                tf.summary.scalar(f'Train/MovingAvgSparse_{layout}', layout_moving_avg_sparse[layout], step=ep)
            summary_writer.flush()

        # --- CHECKPOINT & BEST MODEL ONLY EVERY 1000 ---
        if (ep + 1) % 1000 == 0:
            agent.policy.save_weights(os.path.join(checkpoint_dir, f"policy_{tag}_ep{ep+1}.weights.h5"))
            agent.value.save_weights(os.path.join(checkpoint_dir, f"value_{tag}_ep{ep+1}.weights.h5"))
            print(f"Saved checkpoint at episode {ep+1}")
            if eval_reward_mean > best_eval:
                best_eval = eval_reward_mean
                agent.policy.save_weights(os.path.join(checkpoint_dir, f"policy_{tag}_best.weights.h5"))
                agent.value.save_weights(os.path.join(checkpoint_dir, f"value_{tag}_best.weights.h5"))
                print(f"Saved best model at episode {ep+1} (eval_reward={best_eval:.2f})")

    # --- Final save if not stopped early ---
    agent.policy.save_weights(os.path.join(checkpoint_dir, f"policy_{tag}_final.weights.h5"))
    agent.value.save_weights(os.path.join(checkpoint_dir, f"value_{tag}_final.weights.h5"))
    print("\nFinal model saved.")

    evaluate_policy(agent, env, greedy=rollout_greedy, max_steps=max_steps, save_gif=True, gif_path=gif_path)

if __name__ == '__main__':
    train()

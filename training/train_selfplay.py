import os
import csv
import numpy as np
import tensorflow as tf
import datetime
from env.generalized_env import GeneralizedOvercooked
from agents.ppo_tf import PPOAgent
from training.evaluate_selfplay import evaluate_policy

def train(
    experiment_name="new_setup_15000",
    layout_name='cramped_room',
    network='deep',
    use_shaping=True,
    use_decay=True,
    total_episodes=15000,
    max_steps=400,
    resume=False,
    resume_episode=6000,
    resume_tag = "PPO_deep_shaping-decay-6000_new_rl_final.weights.h5",  # senza ".weights.h5"
    rollout_greedy=True,
):
    shaping_tag = "shaping" if use_shaping else "noshaping"
    decay_tag = "decay" if use_decay else "nodecay"
    tag = f"PPO_{network}_{shaping_tag}-{decay_tag}-{total_episodes}{experiment_name}"

    log_dir = os.path.join(f"{layout_name}/logs")
    checkpoint_dir = os.path.join(f"{layout_name}/checkpoints")
    rollout_dir = os.path.join(f"{layout_name}/rollouts")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(rollout_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"train_log_{tag}.csv")
    gif_path = os.path.join(rollout_dir, f"final_rollout_{tag}.gif")

    tb_log_dir = os.path.join("tensorboard_logs", f"{layout_name}_{tag}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    summary_writer = tf.summary.create_file_writer(tb_log_dir)
    print(f"[TensorBoard] Logging at {tb_log_dir}")

    env = GeneralizedOvercooked([layout_name])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    print("\nTraining PPO agent on Overcooked")
    print(f"Layout: {layout_name}, Obs dim: {obs_dim}, Action dim: {act_dim}")

    agent = PPOAgent(
        obs_dim, act_dim,
        arch_variant=network,
        use_reward_shaping=use_shaping,
        use_decay=use_decay,
        clip_ratio=0.15,
        lr=5e-4  
    )
    start_episode = 0
    if resume:
        try:
            dummy_input = tf.random.uniform((1, obs_dim))
            agent.policy(dummy_input)
            agent.value(dummy_input)
            agent.policy.load_weights(os.path.join(checkpoint_dir, f"policy_{resume_tag}"))
            agent.value.load_weights(os.path.join(checkpoint_dir, f"value_{resume_tag}"))
            start_episode = resume_episode
            print(f"Resumed from checkpoint at episode {resume_episode}")
        except Exception as e:
            print("Failed to load checkpoint:", e)

    csv_header = [
        "episode", "mean_reward", "total_reward", "decay_factor",
        "shaped_reward_first_step", "sparse_reward_first_step", "evaluation_reward_greedy",
        "moving_avg_train_reward"
    ]
    if not os.path.exists(log_path):
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    rolling_eval_rewards = []
    best_eval = -float('inf')

    for ep in range(start_episode, total_episodes):
        obs = env.reset()
        obs_batch, act_batch, old_probs, rewards, dones = [], [], [], [], []

        decay_factor = max(0.0, 1.0 - ep / 12000) if use_decay else 1.0

        shaped_rewards, sparse_rewards = [], []
        episode_reward = 0
        first_shaped = None
        first_sparse = None

        for t in range(max_steps):
            a0, p0 = agent.select_action(obs['both_agent_obs'][0])
            a1, _ = agent.select_action(obs['both_agent_obs'][1])
            actions = [a0, a1]

            next_obs, reward_sparse, done, info = env.step(actions)
            shaped = info.get('shaped_r_by_agent', [0, 0])[0]

            if ep < 2500:
                scale = 10.0
            elif ep < 6000:
                scale = 5.0
            else:
                scale = 2.0

            reward = reward_sparse + decay_factor * (shaped * scale) if use_shaping else reward_sparse

            if t == 0:
                first_shaped = shaped
                first_sparse = reward_sparse

            obs_batch.append(obs['both_agent_obs'][0])
            act_batch.append(a0)
            old_probs.append(p0)
            rewards.append(reward)
            dones.append(done)

            shaped_rewards.append(shaped * scale)
            sparse_rewards.append(reward_sparse)

            obs = next_obs
            episode_reward += reward

            if done:
                break

        print(f"Episode {ep+1}/{total_episodes} | Sparse: {sum(sparse_rewards)} | Shaped : {sum(shaped_rewards)} | Total: {episode_reward:.2f}")

        last_value = agent.value(tf.convert_to_tensor([obs['both_agent_obs'][0]], dtype=tf.float32)).numpy()[0, 0]
        returns = agent.compute_returns(rewards, dones, last_value)
        agent.update(obs_batch, act_batch, old_probs, returns)

        with open(log_path, mode="r") as f:
            lines = f.readlines()
        train_rewards = [float(l.split(',')[2]) for l in lines[1:]] if len(lines) > 1 else []
        moving_avg_reward = np.mean(train_rewards[-100:]) if len(train_rewards) >= 100 else np.mean(train_rewards) if train_rewards else episode_reward

        eval_reward = 0.0
        if (ep + 1) % 100 == 0:
            eval_reward = evaluate_policy(agent, env, greedy=True)
            rolling_eval_rewards.append(eval_reward)
            if len(rolling_eval_rewards) > 10:
                rolling_eval_rewards.pop(0)
            mean_eval = np.mean(rolling_eval_rewards)

            if mean_eval >= 50:
                print(f"\n✅ Early stopping at episode {ep+1}, mean greedy eval (last 10): {mean_eval:.2f}")
                agent.policy.save_weights(os.path.join(checkpoint_dir, f"policy_{tag}_final.weights.h5"))
                agent.value.save_weights(os.path.join(checkpoint_dir, f"value_{tag}_final.weights.h5"))
                evaluate_policy(agent, env, greedy=rollout_greedy, max_steps=max_steps, save_gif=True, gif_path=gif_path)
                print("Training stopped early (target reached!)")
                # Best model always updated at final
                if eval_reward > best_eval:
                    best_eval = eval_reward
                    agent.policy.save_weights(os.path.join(checkpoint_dir, f"policy_{tag}_best.weights.h5"))
                    agent.value.save_weights(os.path.join(checkpoint_dir, f"value_{tag}_best.weights.h5"))
                    print(f"Saved best model at episode {ep+1} (eval_reward={best_eval:.2f})")
                return

        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep, np.mean(rewards), episode_reward, decay_factor,
                first_shaped, first_sparse, eval_reward, moving_avg_reward
            ])

        with summary_writer.as_default():
            tf.summary.scalar('Train/TotalReward', episode_reward, step=ep)
            tf.summary.scalar('Train/MeanReward', np.mean(rewards), step=ep)
            tf.summary.scalar('Train/DecayFactor', decay_factor, step=ep)
            tf.summary.scalar('Train/ShapedFirst', first_shaped, step=ep)
            tf.summary.scalar('Train/SparseFirst', first_sparse, step=ep)
            if 'moving_avg_reward' in locals():
                tf.summary.scalar('Train/MovingAvgReward', moving_avg_reward, step=ep)
            if eval_reward > 0:
                tf.summary.scalar('Eval/GreedyEvalReward', eval_reward, step=ep)
            summary_writer.flush()

        # --- CHECKPOINT & BEST MODEL ONLY EVERY 1000 ---
        if (ep + 1) % 1000 == 0:
            agent.policy.save_weights(os.path.join(checkpoint_dir, f"policy_{tag}_ep{ep+1}.weights.h5"))
            agent.value.save_weights(os.path.join(checkpoint_dir, f"value_{tag}_ep{ep+1}.weights.h5"))
            print(f"Saved checkpoint at episode {ep+1}")
            # best model
            if eval_reward > best_eval:
                best_eval = eval_reward
                agent.policy.save_weights(os.path.join(checkpoint_dir, f"policy_{tag}_best.weights.h5"))
                agent.value.save_weights(os.path.join(checkpoint_dir, f"value_{tag}_best.weights.h5"))
                print(f"Saved best model at episode {ep+1} (eval_reward={best_eval:.2f})")

    # --- Final save if not stopped early ---
    agent.policy.save_weights(os.path.join(checkpoint_dir, f"policy_{tag}_final.weights.h5"))
    agent.value.save_weights(os.path.join(checkpoint_dir, f"value_{tag}_final.weights.h5"))
    print("\nFinal model saved.")

    evaluate_policy(agent, env, greedy=rollout_greedy, max_steps=max_steps, save_gif=True, gif_path=gif_path)

# Fine funzione train

if __name__ == '__main__':
    train()

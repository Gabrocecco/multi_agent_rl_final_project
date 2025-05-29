import tensorflow as tf
from env.generalized_env import GeneralizedOvercooked
from agents.ppo_tf import PPOAgent
from training.evaluate_selfplay import evaluate_policy
import numpy as np

test_layouts = [
    "cramped_room",
    "asymmetric_advantages",
    "forced_coordination",
    ]

# test_layouts = [
#     "forced_coordination"
#     ]

policy_weights = "/home/gabro/Desktop/AAS/final_project/overcooked_rl/multilayout_checkpoints/policy_PPO_deep_shaping-decay-20000multilayout_run_20000_final.weights.h5"
value_weights  = "/home/gabro/Desktop/AAS/final_project/overcooked_rl/multilayout_checkpoints/value_PPO_deep_shaping-decay-20000multilayout_run_20000_final.weights.h5"

results = []

# for layout in test_layouts:
#     print(f"\nEvaluating on layout: {layout}")
#     env = GeneralizedOvercooked([layout])
#     obs_dim = env.observation_space.shape[0]
#     act_dim = env.action_space.n

#     agent = PPOAgent(obs_dim, act_dim, arch_variant='deep', use_reward_shaping=True, use_decay=True)
#     # Dummy forward pass (build weights)
#     agent.policy(tf.random.uniform((1, obs_dim)))
#     agent.value(tf.random.uniform((1, obs_dim)))
#     agent.policy.load_weights(policy_weights)
#     agent.value.load_weights(value_weights)

#     reward = evaluate_policy(agent, env, greedy=True, save_gif=True, gif_path=f"generalization/{layout}/eval.gif")
#     results.append((layout, reward))
#     print(f"Greedy reward on {layout}: {reward}")

# # --- Print summary table ---
# print("\nGeneralization Results (Greedy Evaluation):")
# print("{:<25} | {:>8}".format("Layout", "Reward"))
# print("-" * 36)
# for layout, reward in results:
#     print("{:<25} | {:>8.2f}".format(layout, reward))

N_RUNS = 10
results = []
for layout in test_layouts:
    print(f"\nEvaluating (sampling) on layout: {layout}")
    env = GeneralizedOvercooked([layout])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = PPOAgent(obs_dim, act_dim, arch_variant='deep', use_reward_shaping=True, use_decay=True)
    agent.policy(tf.random.uniform((1, obs_dim)))
    agent.value(tf.random.uniform((1, obs_dim)))
    agent.policy.load_weights(policy_weights)
    agent.value.load_weights(value_weights)

    rewards = []
    for i in range(N_RUNS):
        gif_path=f"generalization/{layout}/eval_sampling_20k_non_greedy.gif" if i == 0 else None
        reward = evaluate_policy(agent, env, greedy=False, save_gif=(i==0), gif_path=gif_path)
        rewards.append(reward)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"Sampling eval on {layout}: mean={mean_reward:.2f} ± {std_reward:.2f}")
    results.append((layout, mean_reward, std_reward))

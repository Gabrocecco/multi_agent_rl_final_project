import imageio
import numpy as np
import tensorflow as tf

# Runs a policy in the given environment for a single episode and optionally saves a GIF of the rollout.
def evaluate_policy(agent, env, greedy=True, max_steps=400, save_gif=False, gif_path="rollout.gif", verbose=False):

    obs = env.reset()
    total_reward = 0
    frames = []
    done = False
    steps = 0

    for t in range(max_steps):
        # Extract the individual observations for both agents
        obs0 = obs['both_agent_obs'][0]
        obs1 = obs['both_agent_obs'][1]

        # === Action selection: Greedy (deterministic) or Sampling (stochastic) ===
        if greedy:
            # Use argmax to select the action with the highest probability
            logits0 = agent.policy(tf.convert_to_tensor([obs0], dtype=tf.float32))
            logits1 = agent.policy(tf.convert_to_tensor([obs1], dtype=tf.float32))
            a0 = tf.argmax(logits0[0]).numpy()
            a1 = tf.argmax(logits1[0]).numpy()
        else:
            # Sample an action according to the policy's probability distribution
            a0, _ = agent.select_action(obs0)
            a1, _ = agent.select_action(obs1)

        # Take a step in the environment with both agents' actions
        obs, reward, done, info = env.step([a0, a1])
        total_reward += reward
        steps += 1

        # === Render and store frame for GIF ===
        if save_gif:
            try:
                frame = None
                if hasattr(env.envs[0], "render"):
                    # Try standard render method (returns RGB array)
                    frame = env.envs[0].render()
                if frame is None and hasattr(env.envs[0], "renderer"):
                    # Fallback: extract RGB from pygame surface
                    surface = getattr(env.envs[0].renderer, "screen", None)
                    if surface is not None:
                        import pygame.surfarray
                        frame = pygame.surfarray.array3d(surface).swapaxes(0, 1)
                if frame is not None:
                    frames.append(frame)
                else:
                    if verbose:
                        print(f"[WARNING] Could not capture frame at step {t}")
            except Exception as e:
                if verbose:
                    print(f"[WARNING] Exception during rendering: {e}")

        if done:
            break

    # === Save GIF if requested ===
    if save_gif and frames:
        try:
            imageio.mimsave(gif_path, frames, fps=10)
            if verbose or True:
                print(f"GIF saved at: {gif_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save GIF: {e}")

    # === Print summary info ===
    if verbose or True:
        print(f"[Eval] Total reward: {total_reward:.2f} | Steps: {steps} | Done: {done}")

    # Optionally, you could return more info (steps, done, etc.)
    return total_reward

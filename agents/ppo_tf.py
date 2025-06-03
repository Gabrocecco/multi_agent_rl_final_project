import tensorflow as tf
import numpy as np

# Fully connected network, "deep" vairiant actually used in the project.
# Two (almost) identical newforks: one for the policy, one for the value fucntion,
# only differeing is the output layer size, which is the action space size for the policy (6 possible actions)
# and 1 for the values funzion (the value of the state). 

class PolicyNetwork(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, variant='baseline'):
        super().__init__()
        if variant == 'baseline':
            # Simple MLP: 1 hidden layer
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(act_dim)
            ])
        elif variant == 'deep':
            # Deeper MLP: 2 hidden layers
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(act_dim)
            ])
        elif variant == 'normalized':
            # MLP with normalization
            self.model = tf.keras.Sequential([
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(act_dim)
            ])
        else:
            raise ValueError(f"Unknown architecture variant: {variant}")

    # Forward pass through the network. 
    # Inputs are the observations, output are the logits for each action.
    def call(self, inputs):
        return self.model(inputs)

class ValueNetwork(tf.keras.Model):
    def __init__(self, obs_dim, variant='baseline'):
        super().__init__()
        if variant == 'baseline':
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
        elif variant == 'deep':
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
        elif variant == 'normalized':
            self.model = tf.keras.Sequential([
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
        else:
            raise ValueError(f"Unknown architecture variant: {variant}")

    # Forward pass though the network. 
    # Inputs are the observations, output is the value of the state.
    def call(self, inputs):
        return self.model(inputs)


# --- PPO Agent ---

class PPOAgent:
    # changed from clip_ratio=0.20 to clip_ratio=0.15
    def __init__(self,
                obs_dim,            # Dimension of the observation space
                act_dim,            # Dimension of the action space (6 possibble actions)
                gamma=0.99,         # Discount factor for future rewards
                clip_ratio=0.15,    # Epsilon for the clipping function
                lr=5e-4,            # Learning rate for the optimizer
                arch_variant='deep',
                use_reward_shaping=True,
                use_decay=True
                ):
        self.gamma = gamma  
        self.clip_ratio = clip_ratio  
        self.use_reward_shaping = use_reward_shaping 
        self.use_decay = use_decay  

        self.policy = PolicyNetwork(obs_dim, act_dim, variant=arch_variant)
        self.value = ValueNetwork(obs_dim, variant=arch_variant)

        # Adam optimizer shared between policy and value networks
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        print(f"\nInitialized PPO agent with architecture: {arch_variant}")
        print(f"obs_dim={obs_dim}, act_dim={act_dim}, gamma={gamma}, clip_ratio={clip_ratio}, lr={lr}")
        print(f"use_reward_shaping={use_reward_shaping}, use_decay={use_decay}")

    # Select an action based on the current observation.
    # The action is sampled from the policy network's output logits.
    def select_action(self, obs):
        obs = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)  # Convert to batch of shape [1, obs_dim]
        logits = self.policy(obs)  # Compute logits from the policy network (before softmax)
        probs = tf.nn.softmax(logits)  # Compute action probabilities via softmax
        action = tf.random.categorical(logits, 1)[0, 0].numpy()  # Sample an action from the logits (non greedy sampling)
        prob = probs[0, action].numpy()  # Probability associated with the chosen action
        return action, prob  # Return action and its probability (needed for PPO loss)

    # For each state visited during the episode, this function calculates the sum of all future rewards
    # (discounted by gamma) that the agent will receive starting from that timestep until the end of the episode.
    def compute_returns(self, rewards, dones, last_value):
        returns = []
        R = last_value 
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1. - d)  # If done == 1, reset return; otherwise, accumulate
            returns.insert(0, R)  # Insert at the front to restore correct order
        return returns  

    # Perform a single PPO update step for both policy and value networks.

    # This function implements the core Proximal Policy Optimization (PPO) update. Given a batch of trajectories
    # collected with the current policy, it:
    #   - Computes the new action probabilities and the ratio to the old policy.
    #   - Applies the PPO clipped surrogate objective to stabilize training.
    #   - Updates both the policy (actor) and the value function (critic) in a single backward pass.

    # PPO update summary:
    #     1. For each (state, action) pair in the batch:
    #         - Calculate current policy probability π_θ(a|s)
    #         - Use stored old policy probability π_θ_old(a|s)
    #         - Compute the probability ratio:
    #             r(θ) = π_θ(a|s) / π_θ_old(a|s)
    #         - Compute the empirical advantage:
    #             A_t = G_t - V_θ(s_t)
    #               where G_t is the return-to-go for timestep t

    #     2. Policy loss (PPO-clip objective):
    #         L_clip(θ) = E_t [ min(r(θ) * A_t, clip(r(θ), 1 - ε, 1 + ε) * A_t) ]
    #         This prevents large, destabilizing updates to the policy.

    #     3. Value loss:
    #         L_vf(θ) = MSE(G_t, V_θ(s_t))

    #     4. Entropy bonus (for exploration):
    #         S[π_θ](s) = -Σ_a π_θ(a|s) log π_θ(a|s)

    #     5. Total loss:
    #         L_total = L_clip + c1 * L_vf - c2 * S[π_θ]
    def update(self,
            obs_batch,   # Batch of observations (states) from the environment
            act_batch,   # Batch of actions taken in those states
            old_probs,   # Old action probabilities from the previous policy (used for PPO ratio)
            returns  # Batch of returns (discounted future rewards) for the states in obs_batch
            ):
        
        # Convert all input data to tensors with appropriate types
        obs_batch = tf.convert_to_tensor(obs_batch, dtype=tf.float32)
        act_batch = tf.convert_to_tensor(act_batch, dtype=tf.int32)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        with tf.GradientTape() as tape:
            logits = self.policy(obs_batch)  # Compute policy logits for all observations
            probs = tf.nn.softmax(logits)    # Compute action probabilities via softmax
            action_probs = tf.gather(probs, act_batch[:, None], batch_dims=1)  # Get the probabilities of the actions actually taken

            # Compute the PPO ratio between new and old action probabilities
            ratio = action_probs[:, 0] / old_probs
            # Apply clipping to the ratio
            clip_adv = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            # Compute empirical advantage (returns - value prediction)
            advantage = returns - tf.squeeze(self.value(obs_batch), axis=1)
            # PPO-CLIP policy loss: minimum between unclipped and clipped objective
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage, clip_adv * advantage))

            # Value loss: mean squared error between returns and value predictions
            value_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(self.value(obs_batch), axis=1)))

            # Entropy bonus: encourages exploration by penalizing certainty
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1))
            entropy_coeff = 0.01  # Weight for entropy bonus (set to 0.0 for no exploration bonus)

            # Total loss: weighted sum of policy loss, value loss, and entropy bonus
            loss = policy_loss + 0.5 * value_loss - entropy_coeff * entropy

        # Compute gradients with respect to all trainable parameters (policy and value networks)
        grads = tape.gradient(loss, self.policy.trainable_variables + self.value.trainable_variables)
        # Apply gradients using Adam optimizer
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables + self.value.trainable_variables))


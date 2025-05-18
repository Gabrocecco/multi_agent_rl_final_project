import tensorflow as tf
import numpy as np

class ConvPolicyNetwork(tf.keras.Model):
    def __init__(self, input_shape, act_dim):
        super().__init__()
        self.reshape = tf.keras.layers.Reshape((input_shape, 1))  # Convert flat vector into 2D input with 1 channel
        self.conv1d_1 = tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')
        self.conv1d_2 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.out = tf.keras.layers.Dense(act_dim)

    def call(self, x):
        x = self.reshape(x)
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.out(x)

class ConvValueNetwork(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__()
        self.reshape = tf.keras.layers.Reshape((input_shape, 1))
        self.conv1d_1 = tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')
        self.conv1d_2 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.out = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.reshape(x)
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.out(x)

class PolicyNetwork(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, variant='baseline'):
        super().__init__()
        if variant == 'baseline':
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(act_dim)
            ])
        elif variant == 'deep':
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(act_dim)
            ])
        elif variant == 'normalized':
            self.model = tf.keras.Sequential([
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(act_dim)
            ])
        else:
            raise ValueError(f"Unknown architecture variant: {variant}")

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

    def call(self, inputs):
        return self.model(inputs)

class PPOAgent:
    def __init__(self, obs_dim, act_dim, gamma=0.99, clip_ratio=0.2, lr=2.5e-4,
                 arch_variant='baseline', use_reward_shaping=True, use_decay=True):
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.use_reward_shaping = use_reward_shaping
        self.use_decay = use_decay

        if arch_variant == 'conv':
            self.policy = ConvPolicyNetwork(obs_dim, act_dim)
            self.value = ConvValueNetwork(obs_dim)
        else:
            self.policy = PolicyNetwork(obs_dim, act_dim, variant=arch_variant)
            self.value = ValueNetwork(obs_dim, variant=arch_variant)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        print(f"\nInitialized PPO agent with architecture: {arch_variant}")
        print(f"obs_dim={obs_dim}, act_dim={act_dim}, gamma={gamma}, clip_ratio={clip_ratio}, lr={lr}")
        print(f"use_reward_shaping={use_reward_shaping}, use_decay={use_decay}")

    def select_action(self, obs):
        obs = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)
        logits = self.policy(obs)
        probs = tf.nn.softmax(logits)
        action = tf.random.categorical(logits, 1)[0, 0].numpy()
        prob = probs[0, action].numpy()
        return action, prob

    def compute_returns(self, rewards, dones, last_value):
        returns = []
        R = last_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1. - d)
            returns.insert(0, R)
        return returns

    def update(self, obs_batch, act_batch, old_probs, returns):
        obs_batch = tf.convert_to_tensor(obs_batch, dtype=tf.float32)
        act_batch = tf.convert_to_tensor(act_batch, dtype=tf.int32)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        with tf.GradientTape() as tape:
            logits = self.policy(obs_batch)
            probs = tf.nn.softmax(logits)
            action_probs = tf.gather(probs, act_batch[:, None], batch_dims=1)

            ratio = action_probs[:, 0] / old_probs
            clip_adv = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            advantage = returns - tf.squeeze(self.value(obs_batch), axis=1)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage, clip_adv * advantage))

            value_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(self.value(obs_batch), axis=1)))
            loss = policy_loss + 0.5 * value_loss

        grads = tape.gradient(loss, self.policy.trainable_variables + self.value.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables + self.value.trainable_variables))

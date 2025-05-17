import tensorflow as tf
import numpy as np

class PolicyNetwork(tf.keras.Model):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.logits = tf.keras.layers.Dense(act_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.logits(x)  # Output logits (for softmax)

class ValueNetwork(tf.keras.Model):
    def __init__(self, obs_dim):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.v = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.v(x)  # Output state value

class PPOAgent:
    def __init__(self, obs_dim, act_dim, gamma=0.99, clip_ratio=0.2, lr=1e-4):
        self.gamma = gamma
        self.clip_ratio = clip_ratio

        self.policy = PolicyNetwork(obs_dim, act_dim)
        self.value = ValueNetwork(obs_dim)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        print(f"\nInitialized PPO agent with obs_dim={obs_dim}, act_dim={act_dim}, "
              f"gamma={gamma}, clip_ratio={clip_ratio}, lr={lr}")
        # print(f"Policy network: {self.policy.summary()}")
        # print(f"Value network: {self.value.summary()}")
        print("PPO agent initialized successfully.\n")

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

            # Policy loss with clipping
            ratio = action_probs[:, 0] / old_probs
            clip_adv = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            advantage = returns - tf.squeeze(self.value(obs_batch), axis=1)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage, clip_adv * advantage))

            # Value function loss (MSE)
            value_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(self.value(obs_batch), axis=1)))

            # Total loss
            loss = policy_loss + 0.5 * value_loss

        # Apply gradients
        grads = tape.gradient(loss, self.policy.trainable_variables + self.value.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables + self.value.trainable_variables))



import tensorflow as tf
import numpy as np

# --- Network basate su convoluzione, utili se vuoi estrarre pattern locali dalle osservazioni ---

class ConvPolicyNetwork(tf.keras.Model):
    def __init__(self, input_shape, act_dim):
        super().__init__()
        # Reshape l'input vettoriale [batch, input_shape] in [batch, input_shape, 1] per usare Conv1D
        self.reshape = tf.keras.layers.Reshape((input_shape, 1))
        # Primo layer convoluzionale 1D con 32 filtri, kernel di ampiezza 5, attivazione ReLU
        self.conv1d_1 = tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')
        # Secondo layer convoluzionale 1D con 64 filtri, kernel di ampiezza 3, attivazione ReLU
        self.conv1d_2 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')
        # Appiattisce l'output (flatten) per connetterlo al fully connected
        self.flatten = tf.keras.layers.Flatten()
        # Densa fully connected con 128 unità e attivazione ReLU
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        # Layer di output: un neurone per ogni possibile azione
        self.out = tf.keras.layers.Dense(act_dim)

    def call(self, x):
        # Definisce il forward pass: reshape -> conv1 -> conv2 -> flatten -> dense -> output
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
        # Output a singolo valore: stima V(s)
        self.out = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.reshape(x)
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.out(x)

# --- Network fully connected (MLP), con possibili varianti di profondità e normalizzazione ---

class PolicyNetwork(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, variant='baseline'):
        super().__init__()
        if variant == 'baseline':
            # MLP semplice: 1 hidden layer 128 -> output
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(act_dim)
            ])
        elif variant == 'deep':
            # MLP più profondo: 256 -> 128 -> output
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(act_dim)
            ])
        elif variant == 'normalized':
            # MLP profondo + layer di normalizzazione
            self.model = tf.keras.Sequential([
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(act_dim)
            ])
        else:
            raise ValueError(f"Unknown architecture variant: {variant}")

    def call(self, inputs):
        # Forward pass: input -> sequenza di layer
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

# --- PPO Agent ---

class PPOAgent:
    # changed from clip_ratio=0.20 to clip_ratio=0.15
    def __init__(self, obs_dim, act_dim, gamma=0.99, clip_ratio=0.15, lr=5e-4,
                 arch_variant='deep', use_reward_shaping=True, use_decay=True):
        self.gamma = gamma  # Discount factor per i reward futuri
        self.clip_ratio = clip_ratio  # Epsilon per la funzione di clipping
        self.use_reward_shaping = use_reward_shaping  # Flag per shaping reward
        self.use_decay = use_decay  # Flag per eventuale learning rate decay

        # Inizializzazione delle reti, conv o fully connected
        if arch_variant == 'conv':
            self.policy = ConvPolicyNetwork(obs_dim, act_dim)
            self.value = ConvValueNetwork(obs_dim)
        else:
            self.policy = PolicyNetwork(obs_dim, act_dim, variant=arch_variant)
            self.value = ValueNetwork(obs_dim, variant=arch_variant)

        # Adam optimizer condiviso tra policy e value
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        print(f"\nInitialized PPO agent with architecture: {arch_variant}")
        print(f"obs_dim={obs_dim}, act_dim={act_dim}, gamma={gamma}, clip_ratio={clip_ratio}, lr={lr}")
        print(f"use_reward_shaping={use_reward_shaping}, use_decay={use_decay}")

    def select_action(self, obs):
        # Prende in input una singola osservazione (dimensione [obs_dim])
        obs = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)  # La converte in batch di shape [1, obs_dim]
        logits = self.policy(obs)  # Calcola logits della policy (prima del softmax)
        probs = tf.nn.softmax(logits)  # Probabilità delle azioni tramite softmax
        action = tf.random.categorical(logits, 1)[0, 0].numpy()  # Campiona un'azione dalle logits (distribuzione categoriale)
        prob = probs[0, action].numpy()  # Probabilità associata all'azione scelta
        return action, prob  # Restituisce azione e probabilità (serve per la loss PPO)

    def compute_returns(self, rewards, dones, last_value):
        # Calcola i return-to-go (reward scontati), propagando da fine episodio a inizio
        returns = []
        R = last_value  # Valore stimato dello stato terminale (per bootstrapping)
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1. - d)  # Se done=1, resetta return
            returns.insert(0, R)  # Inserisce in testa (ricostruendo ordine corretto)
        return returns

    def update(self, obs_batch, act_batch, old_probs, returns):
        # Preprocessamento batch
        obs_batch = tf.convert_to_tensor(obs_batch, dtype=tf.float32)
        act_batch = tf.convert_to_tensor(act_batch, dtype=tf.int32)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        with tf.GradientTape() as tape:
            logits = self.policy(obs_batch)  # Calcola logits per tutte le osservazioni
            probs = tf.nn.softmax(logits)  # Softmax per ottenere distribuzione azioni
            action_probs = tf.gather(probs, act_batch[:, None], batch_dims=1)  # Probabilità delle azioni effettivamente scelte

            # Calcola la ratio tra probabilità nuova e vecchia per PPO
            ratio = action_probs[:, 0] / old_probs
            # Applica il clipping
            clip_adv = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            # Calcola l'advantage empirico (return - value)
            advantage = returns - tf.squeeze(self.value(obs_batch), axis=1)
            # Loss PPO-CLIP: minimo tra ratio*advantage e clip*advantage
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage, clip_adv * advantage))

            # Value loss: errore quadratico tra returns e value stimato
            value_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(self.value(obs_batch), axis=1)))

            # Entropy bonus: incentiva policy esplorative
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1))
            entropy_coeff = 0.01  # Peso del termine di entropia (exploration bonus)
            # Entropy coeff = 0.0 vuol dire no exploration bonus

            # Somma pesata di tutte le componenti della loss
            loss = policy_loss + 0.5 * value_loss - entropy_coeff * entropy

        # Calcola i gradienti rispetto a tutti i parametri (policy e value)
        grads = tape.gradient(loss, self.policy.trainable_variables + self.value.trainable_variables)
        # Applica i gradienti usando Adam
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables + self.value.trainable_variables))

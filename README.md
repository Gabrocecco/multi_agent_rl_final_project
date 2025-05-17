# Overcooked-RL: Cooperative Multi-Agent RL in Overcooked-AI

This project explores layout and partner generalization using PPO (Proximal Policy Optimization) with TensorFlow in the Overcooked-AI benchmark.

---

## 🛠️ Setup Instructions

### 1. Clone Overcooked-AI (required dependency)

```bash
git clone https://github.com/HumanCompatibleAI/overcooked_ai.git
cd overcooked_ai

conda create -n overcooked-rl python=3.10 -y
conda activate overcooked-rl

pip install -e .[harl]

cd path/to/your/overcooked_rl
pip install -r requirements.txt

cd path/to/your/overcooked_rl
python -m training.train_selfplay


overcooked_rl/
│
├── env/                      # Multi-layout wrapper (Gym-like)
│   └── generalized_env.py
├── agents/                   # Custom PPO agent (TensorFlow)
│   └── ppo_tf.py
├── training/                 # Training scripts (e.g., self-play, generalization)
│   └── train_selfplay.py
├── notebooks/                 # Jupyter notebooks for debugging, visualization
│   └── overcooked_starter.ipynb
├── requirements.txt
└── README.md
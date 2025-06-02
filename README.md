# Overcooked-RL: Cooperative Multi-Agent RL in Overcooked-AI

This project explores layout and partner generalization using PPO (Proximal Policy Optimization) with TensorFlow in the Overcooked-AI benchmark.

---

## Setup Instructions

### 1. Clone Overcooked-AI (required dependency)
The Overcooked-AI Python package is needed for all environments and simulation logic.

```bash
git clone https://github.com/HumanCompatibleAI/overcooked_ai.git
cd overcooked_ai
```
### 2. Create and activate a dedicated environment
Recommended to avoid version conflicts.
If you don't use conda, you can use python3.10 -m venv overcooked-rl.
```bash
conda create -n overcooked-rl python=3.10 -y
conda activate overcooked-rl
```

### 3. Install Overcooked-AI (with Human-AI RL extras)
The [harl] option installs all dependencies needed for RL experiments (including gym, pygame, etc).
```bash
pip install -e .[harl]
```

### 4. Install Overcooked-RL requirements
This will install all Python dependencies for this repository (TensorFlow, numpy, etc).
```bash
cd path/to/your/overcooked_rl
pip install -r requirements.txt
```
### 5. Run a training script
Run the `demo.ipynb` notebook for evaluating the project model trained with gifs replays. 


### 6. Run a training script

Now you can start training with PPO using the provided script.
Change script parameters as needed inside `training/train_selfplay.py`.
```bash
cd path/to/your/overcooked_rl
python -m training.train_selfplay
```

overcooked_rl/
│
├── env/
│   └── generalized_env.py          # Multi-layout Gym-like environment wrapper
│
├── agents/
│   └── ppo_tf.py                   # Custom PPO agent (TensorFlow implementation)
│
├── training/
│   └── train_selfplay.py           # Main training script (self-play, generalization)
│
├── notebooks/
│   └── overcooked_starter.ipynb    # Demo & visualization Jupyter notebook
│
├── requirements.txt                # Package requirements
└── README.md                       # Project documentation
# Overcooked-RL: Cooperative Multi-Agent Reinforcement Learning

This project contains PPO-based reinforcement learning experiments on the [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai) benchmark, focusing on cooperation and generalization across multiple kitchen layouts.

---

## 🛠️ Environment Setup

We recommend using **Conda** to create a clean environment with Python 3.10, which is required for the `overcooked_ai` package.

### 1. Clone the Overcooked-AI environment

```bash
git clone https://github.com/HumanCompatibleAI/overcooked_ai.git
cd overcooked_ai

conda create -n overcooked-rl python=3.10 -y
conda activate overcooked-rl

pip install -e .[harl]

cd ../your-overcooked-rl-project
pip install -r requirements.txt
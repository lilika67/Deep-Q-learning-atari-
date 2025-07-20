#  Deep Q-Learning for Atari Games

This repository contains an implementation of Deep Q-Learning (DQN) using **Stable Baselines3** and **Gymnasium** to train and evaluate a reinforcement learning agent in an Atari environment. The agent learns an optimal policy for playing the selected game through deep Q-learning.

---

## Project Structure

```bash
dqn-atari-agent/
├── Scripts/
│   ├── train.py      # Script to train the DQN agent
│   ├── play.py       # Script to evaluate the trained agent
├── models/           # Saved models (e.g., dqn_model.zip)
└── README.md
```

##  Hyperparameter Tuning and Results

Several hyperparameter configurations were tested to optimize the DQN agent's performance. The table below summarizes the experimental setups and observed behavior.

### Hyperparameter Configurations

| Learning Rate (lr) | Gamma (γ) | Batch Size | Epsilon (start → end, decay) | Observed Behavior                                                                                                                                                         |
| ------------------ | --------- | ---------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1e-4               | 0.99      | 32         | 1.0 → 0.05, decay = 0.1      | After 200k steps: Mean reward = **-21** (agent lost every point). Loss stabilized at **0.0008** (still high). Episode length: **762 steps**. Exploration ended at **5%**. |
| 1e-3               | 0.98      | 128         | 1.0 → 0.01, decay = 0.3      | After 500,000 steps: Mean evaluation reward reached -7.40 ± 4.80, showing improved gameplay performance. The agent achieved an average episode length of 12,269 steps ± 550, indicating longer survival. The model checkpoint with best performance was saved at models/dqn_pong_config_1.zip. During training, the average reward stabilized around -17.60, with a final loss of approximately 0.0023. Exploration concluded at 1%, and a total of 122,499 updates were performed. |



> **Policy Comparison**:  
> - **CNNPolicy** consistently outperformed **MLPPolicy** on visual-based Atari environments.  
> - **MLPPolicy** failed to converge on pixel input; best suited for low-dimensional state spaces.

---

##  Agent Demo

> *Include a video here demonstrating your agent playing the game, e.g., as a linked MP4 or GIF.*

- **Video Link**: [[Watch Agent Play Atari Game](https://youtu.be/ibV23DYnSFk)](#)  
- **Script**: `play.py` uses the trained policy with `GreedyQPolicy` to render the agent’s real-time performance.

---

## Team Contributions

| Team Member            | Responsibilities                                                                 |
|------------------------|----------------------------------------------------------------------------------|
| **Sifa Mwachoni**      | Developed `train.py`, implemented CNN/MLP policy comparison, conducted tuning experiments |
| **Liliane Kayitesi**   | Created `play.py`|
| **Denys Ntwaritangazwa** | conducted tuning experiments , analyzed agent performance, and documented findings  |

---

##  How to Run

### 1. Install Dependencies

```bash
pip install stable-baselines3[extra] gymnasium[atari] torch
python src/train.py
python src/play.py


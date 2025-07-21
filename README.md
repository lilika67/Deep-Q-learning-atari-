#  Deep Q-Learning for Atari Games

This repository contains an implementation of Deep Q-Learning (DQN) using **Stable Baselines3** and **Gymnasium** to train and evaluate a reinforcement learning agent in an Atari environment. The agent learns an optimal policy for playing the selected game through deep Q-learning.We focused on training an agent to play **Pong** using the **Arcade Learning Environment (ALE)**.


---


##  Project Structure

```
Deep-Q-learning-atari-/
├── Scripts/                    # Directory for training and saved scripts
│   ├── train.py               # Script to train the agent
│   └── play.py                # Script to play the trained agent
│   └── train-config2            # training with different configurations
├── .gitignore                 # file to ignore on github
├── requirements.tx           # Python dependencies 
└── README.md                 # Project overview and instructions
```


##  Installation

### 1. Clone the repository:
```bash
git clone https://github.com/lilika67/Deep-Q-learning-atari-.git
cd Deep-Q-learning-atari-
```

### 2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  
```

### 3. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 4. Install Atari environments and dependencies:
```bash
pip install "gymnasium[atari,accept-rom-license]"
pip install ale-py autorom
autorom --accept-license
```

---



##  Hyperparameter Tuning and Results

 The table below summarizes the experimental setups and observed behavior.

### Hyperparameter Configurations

# DQN Pong Hyperparameter Tuning Results

| Learning Rate (lr) | Gamma (γ) | Batch Size | Epsilon (Start → End, Decay) | Observed Behavior |
|--------------------|-----------|------------|-------------------------------|--------------------|
| 1e-4               | 0.99      | 32         | 1.0 → 0.05, decay = 0.1       | After 200k steps: Mean reward = -21 (agent lost every point). Loss stabilized at 0.0008 (still high). Episode length: 762 steps. Exploration ended at 5%. |
| 1e-3               | 0.98      | 128        | 1.0 → 0.1, decay = 0.3        | After 500,000 steps: Mean evaluation reward reached -7.40 ± 4.80, showing improved gameplay performance. The agent achieved an average episode length of 12,269 steps ± 550, indicating longer survival. The model checkpoint with best performance was saved at `models/dqn_pong_config_1.zip`. During training, the average reward stabilized around -17.60, with a final loss of approximately 0.0023. Exploration concluded at 1%, and a total of 122,499 updates were performed. |
| 1e-4               | 0.99      | 32         | 1.0 → 0.01, decay = 0.1       | After 100k steps: Mean reward = -21.00 ± 0.00 . Loss stabilized at 0.00698 (relatively high). Episode length: 764 steps ± 0.00, indicating short episodes. Exploration ended at 1%, and a total of 22,499 updates were performed. |


> **Policy Comparison**:  
> - **CNNPolicy** consistently outperformed **MLPPolicy** on visual-based Atari environments.  
> - **MLPPolicy** failed to converge on pixel input; best suited for low-dimensional state spaces.

---

##  Agent Demo
- **Video Link**: [[Watch Agent Play Atari Game](https://youtu.be/ibV23DYnSFk)](#)  
- **Script**: `play.py` uses the trained policy with `GreedyQPolicy` to render the agent’s real-time performance.

---

## Team Contributions

| Team Member            | Responsibilities                                                                 |
|------------------------|----------------------------------------------------------------------------------|
| **Sifa Mwachoni**      | Developed `train.py`, implemented CNN/MLP policy comparison, conducted tuning experiments |
| **Liliane Kayitesi**   | Created the `play.py` script for model evaluation, Implemented model loading functionality with environment consistency checks |
| **Denys Ntwaritangazwa** | conducted tuning experiments , analyzed agent performance, and documented findings  |

---

##  How to Run

### 1. Install Dependencies

```bash
pip install stable-baselines3[extra] gymnasium[atari] torch
python src/train.py
python src/play.py


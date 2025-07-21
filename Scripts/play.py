# play_breakout.py

import ale_py
import gymnasium as gym
import numpy as np
import time

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    WarpFrame,
    ClipRewardEnv
)

# Step 1: Create a custom Atari environment similar to SB3's pipeline
def make_custom_atari_env():
    env = gym.make("PongNoFrameskip-v4", render_mode='human')  # 👁️ Enable GUI rendering
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env

# Step 2: Wrap environment for SB3 with DummyVecEnv and frame stacking
env = DummyVecEnv([make_custom_atari_env])
env = VecFrameStack(env, n_stack=4)

# Step 3: Load the pre-trained model (change name if needed)
model_path = "models/dqn_pong_config_1"  # 
model = DQN.load(model_path)

# Step 4: Evaluate the agent visually
obs = env.reset()
total_reward = 0

for step in range(1000):
    # Predict the next action using the trained model
    action, _states = model.predict(obs, deterministic=True)

    # Step through the environment
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]  # reward is returned as a list due to DummyVecEnv

    # Optional: Add delay for smoother GUI visualization (30 fps = ~33ms per frame)
    time.sleep(1 / 30.0)

    # Log progress every 100 steps
    if step % 100 == 0:
        print(f"Step {step}, Cumulative reward: {total_reward}")

    # Handle episode termination
    if done[0]:
        print(f"🏁 Episode finished at step {step}, Total reward: {total_reward}")
        obs = env.reset()
        total_reward = 0

# Step 5: Clean up
env.close()
print(" Environment closed.")
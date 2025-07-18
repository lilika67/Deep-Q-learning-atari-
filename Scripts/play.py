import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
import ale_py

def make_env():
    gym.register_envs(ale_py)  
    env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=42, env_kwargs={"render_mode": "human"})  # Set render_mode
    env = VecFrameStack(env, n_stack=9)  
    return env

def play_model(model_path):
    env = make_env()
    model = DQN.load(model_path)

    obs = env.reset()
    done = False

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()  
        if done:
            obs = env.reset()

if __name__ == "__main__":
    model_path = "models/dqn_pong.zip"  
    play_model(model_path)
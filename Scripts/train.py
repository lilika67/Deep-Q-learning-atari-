# train.py
import gymnasium as gym
#import gymnasium.envs.atari
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import os

def make_env(env_id, seed=42):
    env = make_atari_env(env_id, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    return env

# Each member uses a unique fine-tuned configuration
hyperparams = [
    {
        'policy': 'CnnPolicy',
        'lr': 1e-4,
        'gamma': 0.99,
        'batch_size': 32,
        'exploration_fraction': 0.1,
        'exploration_final_eps': 0.01,
        'buffer_size': 100000,
        'learning_starts': 10000,
        'target_update_interval': 1000
    },
    {
        'policy': 'CnnPolicy',
        'lr': 5e-4,
        'gamma': 0.97,
        'batch_size': 64,
        'exploration_fraction': 0.15,
        'exploration_final_eps': 0.05,
        'buffer_size': 80000,
        'learning_starts': 8000,
        'target_update_interval': 750
    },
    {
        'policy': 'CnnPolicy',
        'lr': 3e-4,
        'gamma': 0.95,
        'batch_size': 32,
        'exploration_fraction': 0.2,
        'exploration_final_eps': 0.02,
        'buffer_size': 120000,
        'learning_starts': 15000,
        'target_update_interval': 1500
    },
    {
        'policy': 'CnnPolicy',
        'lr': 1e-3,
        'gamma': 0.98,
        'batch_size': 128,
        'exploration_fraction': 0.3,
        'exploration_final_eps': 0.1,
        'buffer_size': 60000,
        'learning_starts': 6000,
        'target_update_interval': 500
    },
]

def train_agent(env_id="PongNoFrameskip-v4", total_timesteps=500000):
    os.makedirs("models", exist_ok=True)
    env = make_env(env_id)
    eval_env = make_env(env_id)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=25000,
        deterministic=True,
        render=False
    )

    for i, config in enumerate(hyperparams):
        print(f"\n Training Config {i+1}: {config}\n")

        model = DQN(
            policy=config['policy'],
            env=env,
            verbose=1,
            tensorboard_log="./tensorboard/",
            learning_rate=config['lr'],
            gamma=config['gamma'],
            batch_size=config['batch_size'],
            exploration_fraction=config['exploration_fraction'],
            exploration_final_eps=config['exploration_final_eps'],
            buffer_size=config['buffer_size'],
            learning_starts=config['learning_starts'],
            target_update_interval=config['target_update_interval'],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            tb_log_name=f"dqn_pong_config_{i+1}"
        )

        model.save(f"models/dqn_pong_config_{i+1}.zip")
        print(f" Model saved: models/dqn_pong_config_{i+1}.zip")

        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print(f" Mean reward for config {i+1}: {mean_reward:.2f}")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    train_agent()

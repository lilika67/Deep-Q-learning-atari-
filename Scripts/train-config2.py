import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import torch
import ale_py
import os


def make_env(env_id, seed=42, n_stack=4):
    env = make_atari_env(env_id, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=n_stack)
    return env

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
        'policy': 'MlpPolicy',
        'lr': 1e-3,
        'gamma': 0.95,
        'batch_size': 64,
        'exploration_fraction': 0.2,
        'exploration_final_eps': 0.02,
        'buffer_size': 50000,
        'learning_starts': 5000,
        'target_update_interval': 500
    },
    {
        'policy': 'CnnPolicy',
        'lr': 5e-5,
        'gamma': 0.98,
        'batch_size': 128,
        'exploration_fraction': 0.15,
        'exploration_final_eps': 0.05,
        'buffer_size': 200000,
        'learning_starts': 20000,
        'target_update_interval': 2000,
        'policy_kwargs': {'net_arch': [512, 256]}
    },
    {
        'policy': 'CnnPolicy',
        'lr': 2e-4,
        'gamma': 0.995,
        'batch_size': 16,
        'exploration_fraction': 0.05,
        'exploration_final_eps': 0.01,
        'buffer_size': 50000,
        'learning_starts': 5000,
        'target_update_interval': 500,
        'policy_kwargs': {'net_arch': [256, 128]}
    },
    {
        'policy': 'CnnPolicy',
        'lr': 1e-4,
        'gamma': 0.99,
        'batch_size': 32,
        'exploration_fraction': 0.1,
        'exploration_final_eps': 0.01,
        'buffer_size': 50000, 
        'learning_starts': 10000,
        'target_update_interval': 1000,
        'env_id': 'ALE/Pong-v5',
        'n_stack': 9
    }
]

def get_model_filename(config_env_id, config_index):
    if config_index >= 2:
        base_name = "pong" if config_env_id == "ALE/Pong-v5" else "breakout"
        return f"models/dqn_{base_name}{config_index-1}.zip"
    return f"models/dqn_{config_env_id}_config_{config_index+1}.zip"

def train_agent(env_id="ALE/Breakout-v5", total_timesteps=100000, config_indices="new"):
    if config_indices == "new":
        configs = hyperparams[2:]
        start_index = 2
    elif config_indices == "all":
        configs = hyperparams
        start_index = 0
    else:
        configs = [hyperparams[i] for i in config_indices]
        start_index = min(config_indices)

    for i, config in enumerate(configs, start=start_index):
        config_env_id = config.get('env_id', env_id)
        config_n_stack = config.get('n_stack', 4)
        env = make_env(config_env_id, n_stack=config_n_stack)
        eval_env = make_env(config_env_id, n_stack=config_n_stack)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./logs/",
            log_path="./logs/",
            eval_freq=20000,  
            deterministic=True,
            render=False
        )
        
        print(f"\nTraining with config {i+1}: {config}\n")
        
        model = DQN(
            config['policy'],
            env,
            verbose=1,
            
            learning_rate=config['lr'],
            gamma=config['gamma'],
            batch_size=config['batch_size'],
            exploration_fraction=config['exploration_fraction'],
            exploration_final_eps=config['exploration_final_eps'],
            buffer_size=config['buffer_size'],
            learning_starts=config['learning_starts'],
            target_update_interval=config['target_update_interval'],
            device="cuda" if torch.cuda.is_available() else "cpu",
            policy_kwargs=config.get('policy_kwargs', {})
        )
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            tb_log_name=f"dqn_{config_env_id}_config_{i+1}"
        )
        
        model_filename = get_model_filename(config_env_id, i)
        model.save(model_filename)
        print(f"Saved model to {model_filename}")
        
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print(f"Mean reward for config {i+1} ({config_env_id}): {mean_reward:.2f}")
        
        env.close()
        eval_env.close()

if __name__ == "__main__":
    ENV_ID = "ALE/Breakout-v5"
    train_agent(env_id=ENV_ID, total_timesteps=100000, config_indices=[4])  
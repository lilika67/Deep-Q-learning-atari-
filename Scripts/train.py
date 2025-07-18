import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import EvalCallback
import torch

def make_env(env_id, seed=42):
    env = make_atari_env(env_id, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
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
]

def train_agent(env_id="ALE/Breakout-v5", total_timesteps=100000):
    env = make_env(env_id)
    eval_env = make_env(env_id)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    for i, config in enumerate(hyperparams):
        print(f"\nTraining with config {i+1}: {config}\n")
        
        model = DQN(
            config['policy'],
            env,
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
            tb_log_name=f"dqn_{env_id}_config_{i+1}"
        )
        
        model.save(f"dqn_{env_id}_config_{i+1}")
        
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print(f"Mean reward for config {i+1}: {mean_reward:.2f}")
        
        env.close()
        eval_env.close()

if __name__ == "__main__":
    ENV_ID = "ALE/Breakout-v5"
    train_agent(env_id=ENV_ID, total_timesteps=1000000)

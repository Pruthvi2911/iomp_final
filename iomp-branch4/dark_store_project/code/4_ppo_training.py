"""
Dark Store Project - PPO Training
Step 4: Train the reinforcement learning agent
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO #PPO Algorithm
from stable_baselines3.common.callbacks import BaseCallback #Callback to track progress
from stable_baselines3.common.vec_env import DummyVecEnv #Vectorized Environment to speed up training
import time #Time tracking
import sys #System specific parameters and functions
from pathlib import Path #Path manipulation

# Fix terminal encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

# ============================================
# CONFIGURATION & PATHS
# ============================================
BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = BASE_DIR / "code"
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LAYOUT_FILE = str(RESULTS_DIR / "optimized_layout.json")
MODEL_SAVE_PATH = str(RESULTS_DIR / "ppo_model")
LOG_PATH = str(RESULTS_DIR / "training_log.csv")

# Dynamic import of your environment
sys.path.append(str(CODE_DIR))
import importlib
try:
    # Handles both '3_gym_environment.py' and 'gym_environment.py'
    gym_module = importlib.import_module("3_gym_environment")
except ImportError:
    gym_module = importlib.import_module("gym_environment")
DarkStoreEnv = gym_module.DarkStoreEnv

# Training parameters
TOTAL_TIMESTEPS = 500000  #Number of steps to train the agent
LEARNING_RATE = 3e-4 #Learning rate 0.0003 controls update size
N_STEPS = 2048 #Number of steps to train the agent
BATCH_SIZE = 64 #Batch size 2^n 
N_EPOCHS = 10 #Number of epochs Gradient update count.

print("=" * 60)
print("DARK STORE - PPO TRAINING")
print("=" * 60)

# ============================================
# CUSTOM CALLBACK - Track Progress
# ============================================
class ProgressCallback(BaseCallback):
    def __init__(self, check_freq=1000, log_path=None, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_path = log_path
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.logs = []
        
    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            info = self.locals['infos'][0]
            # Handle info being inside a list (SB3 Vectorized Env style)
            success = info.get('success', False)
            self.episode_successes.append(1 if success else 0)
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        if self.num_timesteps % self.check_freq == 0:
            if len(self.episode_rewards) > 0:
                recent_success_rate = np.mean(self.episode_successes[-100:]) * 100
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Step: {self.num_timesteps} | Success Rate: {recent_success_rate:.1f}% | Avg Reward: {avg_reward:.2f}")
                self.logs.append({
                    'timestep': self.num_timesteps,
                    'avg_reward': avg_reward,
                    'avg_length': np.mean(self.episode_lengths[-100:]),
                    'success_rate': recent_success_rate
                })
        return True
    
    def _on_training_end(self) -> None:
        if self.log_path and self.logs:
            pd.DataFrame(self.logs).to_csv(self.log_path, index=False)

# ============================================
# EXECUTION
# ============================================
print("\n[1/5] Creating environment...")
env = DarkStoreEnv(LAYOUT_FILE, max_steps=250, max_items_per_order=3)
env = DummyVecEnv([lambda: env])

print("[2/5] Initializing PPO model...")
model = PPO(
    "MlpPolicy", env,
    learning_rate=LEARNING_RATE,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    ent_coef=0.02, # Higher entropy encourages exploration in a maze
    verbose=0
) # PPO Creation with 4 parameters MlpPolicy, Env, Learning Rate, N_STEPS, BATCH_SIZE, N_EPOCHS, ent_coef
# ppo creation Multi-Layer Perceptron.
print("[3/5] Starting training (500k steps)...")
callback = ProgressCallback(log_path=LOG_PATH)
start_time = time.time()
try:
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    print(f"\n✓ Training finished in {(time.time()-start_time)/60:.1f} minutes")
except KeyboardInterrupt:
    print("\n⚠ Interrupted! Saving progress...")

print("[4/5] Saving model...")
model.save(MODEL_SAVE_PATH)

print("[5/5] Generating Learning Curves...")
if os.path.exists(LOG_PATH):
    df = pd.read_csv(LOG_PATH)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df['timestep'], df['success_rate'])
    plt.title('Success Rate (%)')
    plt.subplot(1, 2, 2)
    plt.plot(df['timestep'], df['avg_reward'])
    plt.title('Average Reward')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ppo_learning_curve.png")
    print(f"✓ Visualization saved to {RESULTS_DIR}")

print("=" * 60)
print("PPO TRAINING COMPLETE!")
print("=" * 60)
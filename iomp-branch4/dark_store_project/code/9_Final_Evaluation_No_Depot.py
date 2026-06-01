import sys
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import importlib
from pathlib import Path

# Fix terminal encoding for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Resolve paths
BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = BASE_DIR / "code"
RESULTS_DIR = BASE_DIR / "results"

sys.path.append(str(CODE_DIR))

# Dynamic import of environment
try:
    gym_env_module = importlib.import_module("3_gym_environment")
except ImportError:
    gym_env_module = importlib.import_module("gym_environment")
DarkStoreEnv = gym_env_module.DarkStoreEnv

# File Paths
OPT_LAYOUT_PATH = str(RESULTS_DIR / "optimized_layout.json")
OPT_MODEL_PATH = str(RESULTS_DIR / "ppo_model.zip")

n_eval = 200
SLA_STEPS = 250
EVAL_SEED = 42

print("=" * 80)
print("DARK STORE - EVALUATION (NO DEPOT REQUIREMENT)")
print("Goal: Only requires agent to pick up all items. No return to depot needed.")
print("=" * 80)

# ==================================
# 1. EVALUATE RANDOM AGENT
# ==================================
print(f"\n[1/2] Evaluating Random Walk on ILP Optimized Layout...")
env_opt = DarkStoreEnv(OPT_LAYOUT_PATH, max_steps=SLA_STEPS, max_items_per_order=3)

rand_lengths, rand_successes = [], []
for i in range(n_eval):
    obs, info = env_opt.reset(seed=EVAL_SEED + i)
    done = False
    ep_length = 0
    while not done:
        action = env_opt.action_space.sample() # Agent moves randomly Used as baseline.
        obs, reward, terminated, truncated, info = env_opt.step(action)
        ep_length += 1
        done = terminated or truncated
    
    rand_lengths.append(ep_length)
    # SUCCESS CONDITION CHECK: Only check if items_remaining is 0. Ignore depot.
    items_remaining = info.get('items_remaining', 3)
    rand_successes.append(1 if items_remaining == 0 else 0)

# ==================================
# 2. EVALUATE PPO ON OPTIMIZED LAYOUT
# ==================================
print(f"[2/2] Evaluating PPO Agent (Stochastic) on ILP Optimized Layout...")
model_opt = PPO.load(OPT_MODEL_PATH, device='cpu')

ppo_opt_lengths, ppo_opt_successes = [], []
for i in range(n_eval):
    obs, info = env_opt.reset(seed=EVAL_SEED + i)
    done = False
    ep_length = 0
    while not done:
        # Use stochastic prediction (deterministic=False) to prevent gridlock loops
        action, _ = model_opt.predict(obs, deterministic=False) #uses trained PPO model to select action
        obs, reward, term, trunc, info = env_opt.step(action)
        ep_length += 1
        done = term or trunc
        
    ppo_opt_lengths.append(ep_length)
    # SUCCESS CONDITION CHECK: Only check if items_remaining is 0. Ignore depot.
    items_remaining = info.get('items_remaining', 3)
    ppo_opt_successes.append(1 if items_remaining == 0 else 0) #In no-depot evaluation. Meaning All items picked. Return depot not required.

# ==================================
# 3. COMPILE & SAVE METRICS
# ==================================
def avg_success_steps(lengths, successes):
    success_steps = [l for l, s in zip(lengths, successes) if s == 1]
    return np.mean(success_steps) if success_steps else 0.0

metrics = [
    {
        'Method': 'Random Walk + ILP Layout',
        'Avg Steps': np.mean(rand_lengths),
        'Avg Steps (Win)': avg_success_steps(rand_lengths, rand_successes),
        'Success Rate (%)': np.mean(rand_successes) * 100
    },
    {
        'Method': 'PPO (Stochastic) + ILP Layout',
        'Avg Steps': np.mean(ppo_opt_lengths),
        'Avg Steps (Win)': avg_success_steps(ppo_opt_lengths, ppo_opt_successes),
        'Success Rate (%)': np.mean(ppo_opt_successes) * 100
    }
]

df = pd.DataFrame(metrics)
df.to_csv(RESULTS_DIR / "no_depot_comparison.csv", index=False)

print("\nFINAL EVALUATION RESULTS (JUST PICKING ITEMS, NO DEPOT RETURN REQUIRED):")
print("-" * 85)
print(f"{'Method':<35}| {'Avg Total Steps':<18}| {'Avg Steps (Win)':<18}| {'Success Rate'}")
print("-" * 85)
for _, row in df.iterrows():
    marker = "*BEST*" if "PPO" in row['Method'] else ""
    succ_str = f"{row['Avg Steps (Win)']:.1f}" if row['Avg Steps (Win)'] > 0 else "N/A"
    print(f"{row['Method']:<35}| {row['Avg Steps']:<18.1f}| {succ_str:<18}| {row['Success Rate (%)']:.1f}% {marker}")
print("-" * 85)
print(f"\n✓ Metrics saved to: {RESULTS_DIR}/no_depot_comparison.csv\n")

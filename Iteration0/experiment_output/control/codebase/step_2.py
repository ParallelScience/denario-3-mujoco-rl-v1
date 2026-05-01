# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import os
import time
from step_1 import ActuatorLatencyWrapper, ActionHistoryWrapper
DATA_DIR = "data/"
SEED = 42
LATENCY_LEVELS = [0, 1, 2, 3, 5]
N_EVAL_EPISODES = 5
MAX_DELAY = 5
K_B = 3
K_C = 10
def make_eval_env_A(seed, max_delay=MAX_DELAY):
    base_env = gym.make('HalfCheetah-v4')
    base_env.reset(seed=seed)
    return ActuatorLatencyWrapper(base_env, initial_delay=1, max_delay=max_delay)
def make_eval_env_B(seed, k=K_B, max_delay=MAX_DELAY):
    base_env = gym.make('HalfCheetah-v4')
    base_env.reset(seed=seed)
    return ActionHistoryWrapper(base_env, k=k, initial_delay=1, max_delay=max_delay)
def make_eval_env_C(seed, k=K_C, max_delay=MAX_DELAY):
    base_env = gym.make('HalfCheetah-v4')
    base_env.reset(seed=seed)
    return ActionHistoryWrapper(base_env, k=k, initial_delay=1, max_delay=max_delay)
def extract_x_velocity(info):
    if 'x_velocity' in info:
        val = info['x_velocity']
        if val is not None:
            return float(val)
    if 'final_info' in info:
        fi = info['final_info']
        if isinstance(fi, dict) and 'x_velocity' in fi:
            val = fi['x_velocity']
            if val is not None:
                return float(val)
        if isinstance(fi, (list, np.ndarray)):
            for item in fi:
                if item is not None and isinstance(item, dict) and 'x_velocity' in item:
                    val = item['x_velocity']
                    if val is not None:
                        return float(val)
    return float('nan')
def evaluate_model(model, env, n_episodes=5):
    episode_returns = []
    all_velocities = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=SEED + ep)
        done = False
        ep_return = 0.0
        ep_velocities = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            v = extract_x_velocity(info)
            if not np.isnan(v):
                ep_velocities.append(v)
            done = terminated or truncated
        episode_returns.append(ep_return)
        if ep_velocities:
            all_velocities.extend(ep_velocities)
    mean_return = float(np.mean(episode_returns))
    mean_velocity = float(np.mean(all_velocities)) if all_velocities else float('nan')
    return mean_return, mean_velocity
def smooth_curve(values, window=10):
    values = np.array(values, dtype=np.float64)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - window // 2 - 1), mode='edge')
    return np.convolve(padded, kernel, mode='valid')
if __name__ == '__main__':
    np.random.seed(SEED)
    model_A = SAC.load(os.path.join(DATA_DIR, "sac_condition_A"), device='cuda')
    model_B = SAC.load(os.path.join(DATA_DIR, "sac_condition_B"), device='cuda')
    model_C = SAC.load(os.path.join(DATA_DIR, "sac_condition_C"), device='cuda')
    conditions = [("Condition A (Baseline)", model_A, 'A'), ("Condition B (History-MLP k=3)", model_B, 'B'), ("Condition C (Extended-History k=10)", model_C, 'C')]
    env_factories = {'A': make_eval_env_A, 'B': make_eval_env_B, 'C': make_eval_env_C}
    results = []
    for cond_name, model, cond_key in conditions:
        for latency in LATENCY_LEVELS:
            env = env_factories[cond_key](seed=SEED)
            env.set_delay(latency)
            mean_ret, mean_vel = evaluate_model(model, env, n_episodes=N_EVAL_EPISODES)
            results.append({'condition': cond_name, 'latency': latency, 'mean_return': mean_ret, 'mean_velocity': mean_vel})
    df_results = pd.DataFrame(results)
    results_path = os.path.join(DATA_DIR, "robustness_evaluation.csv")
    df_results.to_csv(results_path, index=False)
    df_A = pd.read_csv(os.path.join(DATA_DIR, "training_curves_A.csv"))
    df_B = pd.read_csv(os.path.join(DATA_DIR, "training_curves_B.csv"))
    df_C = pd.read_csv(os.path.join(DATA_DIR, "training_curves_C.csv"))
    smooth_window = max(5, min(15, len(df_A) // 10))
    ts_A, ret_A = df_A['timestep'].values, smooth_curve(df_A['episode_return'].values, window=smooth_window)
    ts_B, ret_B = df_B['timestep'].values, smooth_curve(df_B['episode_return'].values, window=smooth_window)
    ts_C, ret_C = df_C['timestep'].values, smooth_curve(df_C['episode_return'].values, window=smooth_window)
    colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}
    markers = {'A': 'o', 'B': 's', 'C': '^'}
    labels = {'A': 'Condition A (Baseline)', 'B': 'Condition B (History k=3)', 'C': 'Condition C (Ext-History k=10)'}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(ts_A, ret_A, color=colors['A'], label=labels['A'])
    axes[0].plot(ts_B, ret_B, color=colors['B'], label=labels['B'])
    axes[0].plot(ts_C, ret_C, color=colors['C'], label=labels['C'])
    axes[0].set_xlabel("Training Timesteps")
    axes[0].set_ylabel("Episode Return (smoothed)")
    axes[0].legend()
    df_rob_A = df_results[df_results['condition'] == "Condition A (Baseline)"]
    df_rob_B = df_results[df_results['condition'] == "Condition B (History-MLP k=3)"]
    df_rob_C = df_results[df_results['condition'] == "Condition C (Extended-History k=10)"]
    for i, df_rob in enumerate([df_rob_A, df_rob_B, df_rob_C]):
        key = ['A', 'B', 'C'][i]
        axes[1].plot(df_rob['latency'], df_rob['mean_return'], color=colors[key], marker=markers[key], label=labels[key])
        axes[2].plot(df_rob['latency'], df_rob['mean_velocity'], color=colors[key], marker=markers[key], label=labels[key])
    axes[1].set_xlabel("Latency Level"); axes[1].set_ylabel("Mean Return"); axes[1].legend()
    axes[2].set_xlabel("Latency Level"); axes[2].set_ylabel("Mean Velocity"); axes[2].legend()
    plt.tight_layout()
    plot_path = os.path.join(DATA_DIR, "results_summary_" + str(int(time.time())) + ".png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
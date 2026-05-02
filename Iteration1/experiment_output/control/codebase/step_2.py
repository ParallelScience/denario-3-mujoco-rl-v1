# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import SAC
import warnings
import os
from step_1 import LatencyWrapper, ObservationAugmenter

warnings.filterwarnings('ignore')

DATA_DIR = 'data/'
LATENCY_LEVELS = [0, 1, 2, 3, 5]
N_EVAL_EPISODES = 10
SEED = 42

def run_episodes(model, env, n_episodes=10):
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=SEED + ep)
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        returns.append(total_reward)
    return returns

def make_gold_eval_env(latency):
    base = gym.make('HalfCheetah-v4')
    lat = LatencyWrapper(base, initial_delay=latency)
    return lat

def make_augmented_eval_env(latency, k):
    base = gym.make('HalfCheetah-v4')
    lat = LatencyWrapper(base, initial_delay=latency)
    aug = ObservationAugmenter(lat, k=k)
    return aug

def evaluate_condition(model, make_env_fn, latency_levels, n_episodes):
    results = {}
    for lat in latency_levels:
        env = make_env_fn(lat)
        ep_returns = run_episodes(model, env, n_episodes)
        env.close()
        results[lat] = {'mean': float(np.mean(ep_returns)), 'std': float(np.std(ep_returns))}
    return results

if __name__ == '__main__':
    np.random.seed(SEED)
    gold_model_path = os.path.join(DATA_DIR, 'gold_model.zip')
    cond_a_model_path = os.path.join(DATA_DIR, 'cond_a_model.zip')
    cond_b_model_path = os.path.join(DATA_DIR, 'cond_B_k3_model.zip')
    cond_c_model_path = os.path.join(DATA_DIR, 'cond_C_k10_model.zip')
    gold_model = SAC.load(gold_model_path, device='cuda')
    cond_a_model = SAC.load(cond_a_model_path, device='cuda')
    cond_b_model = SAC.load(cond_b_model_path, device='cuda')
    cond_c_model = SAC.load(cond_c_model_path, device='cuda')
    gold_results = evaluate_condition(gold_model, lambda lat: make_gold_eval_env(lat), LATENCY_LEVELS, N_EVAL_EPISODES)
    cond_a_results = evaluate_condition(cond_a_model, lambda lat: make_augmented_eval_env(lat, k=0), LATENCY_LEVELS, N_EVAL_EPISODES)
    cond_b_results = evaluate_condition(cond_b_model, lambda lat: make_augmented_eval_env(lat, k=3), LATENCY_LEVELS, N_EVAL_EPISODES)
    cond_c_results = evaluate_condition(cond_c_model, lambda lat: make_augmented_eval_env(lat, k=10), LATENCY_LEVELS, N_EVAL_EPISODES)
    conditions = {'Gold Standard (k=0, no latency training)': gold_results, 'Condition A (k=0, padded, curriculum)': cond_a_results, 'Condition B (k=3, history, curriculum)': cond_b_results, 'Condition C (k=10, history, curriculum)': cond_c_results}
    rows = []
    for cond_name, res in conditions.items():
        for lat in LATENCY_LEVELS:
            rows.append({'condition': cond_name, 'latency': lat, 'mean_return': res[lat]['mean'], 'std_return': res[lat]['std']})
    df = pd.DataFrame(rows)
    csv_path = os.path.join(DATA_DIR, 'robustness_results.csv')
    df.to_csv(csv_path, index=False)
    print('Robustness results saved to ' + csv_path)
    print('\n' + '='*80)
    print('ROBUSTNESS EVALUATION TABLE: Mean Return +/- Std across Latency Levels')
    print('='*80)
    header = '{:<45s}'.format('Condition') + ''.join(['{:>14s}'.format('Lat=' + str(l)) for l in LATENCY_LEVELS])
    print(header)
    print('-'*80)
    for cond_name, res in conditions.items():
        row_str = '{:<45s}'.format(cond_name[:44])
        for lat in LATENCY_LEVELS:
            cell = str(round(res[lat]['mean'], 1)) + '+/-' + str(round(res[lat]['std'], 1))
            row_str += '{:>14s}'.format(cell)
        print(row_str)
    print('='*80)
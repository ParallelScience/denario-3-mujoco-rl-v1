# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime

DATA_DIR = "data/"

def normalize_eval_log_columns(df):
    if 'timestep' in df.columns and 'timesteps' not in df.columns:
        df = df.rename(columns={'timestep': 'timesteps'})
    return df

def load_eval_log(path, label):
    df = pd.read_csv(path)
    df = normalize_eval_log_columns(df)
    if 'timesteps' not in df.columns or 'mean_reward' not in df.columns:
        raise ValueError("CSV " + path + " missing required columns after normalization. Found: " + str(list(df.columns)))
    print(label + " eval log loaded: " + str(len(df)) + " rows, steps " + str(int(df['timesteps'].min())) + " to " + str(int(df['timesteps'].max())))
    return df

def load_robustness(path):
    df = pd.read_csv(path)
    print("Robustness CSV loaded: " + str(len(df)) + " rows.")
    print("Conditions found: " + str(df['condition'].unique().tolist()))
    return df

def plot_learning_curves(logs_dict, curriculum_onset_step, save_path):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, (label, df) in enumerate(logs_dict.items()):
        ax.plot(df['timesteps'], df['mean_reward'], color=colors[idx % len(colors)], linestyle=linestyles[idx % len(linestyles)], marker=markers[idx % len(markers)], markersize=4, linewidth=2, label=label)
    ax.axvline(x=curriculum_onset_step, color='black', linestyle='--', linewidth=1.5, label='Curriculum onset (step ' + str(curriculum_onset_step) + ')')
    ax.set_xlabel('Training Steps', fontsize=13)
    ax.set_ylabel('Mean Episode Return', fontsize=13)
    ax.set_title('Learning Curves: HalfCheetah-v4 under Curriculum Latency', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.4)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x // 1000)) + 'k'))
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_robustness_profile(rob_df, gold_zero_latency_return, save_path):
    conditions = rob_df['condition'].unique().tolist()
    latency_levels = sorted(rob_df['latency'].unique().tolist())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, cond in enumerate(conditions):
        sub = rob_df[rob_df['condition'] == cond].sort_values('latency')
        ax.errorbar(sub['latency'], sub['mean_return'], yerr=sub['std_return'], color=colors[idx % len(colors)], marker=markers[idx % len(markers)], linestyle=linestyles[idx % len(linestyles)], linewidth=2, markersize=7, capsize=5, label=cond)
    ax.axhline(y=gold_zero_latency_return, color='gray', linestyle='--', linewidth=1.5, label='Gold Std zero-latency ref (' + str(round(gold_zero_latency_return, 1)) + ')')
    ax.set_xlabel('Latency (steps)', fontsize=13)
    ax.set_ylabel('Mean Episode Return', fontsize=13)
    ax.set_title('Robustness Profile: Mean Return vs. Actuator Latency', fontsize=14)
    ax.set_xticks(latency_levels)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    gold_log_path = os.path.join(DATA_DIR, 'gold_eval_log.csv')
    cond_a_log_path = os.path.join(DATA_DIR, 'cond_A_k0_eval_log.csv')
    cond_b_log_path = os.path.join(DATA_DIR, 'cond_B_k3.csv')
    cond_c_log_path = os.path.join(DATA_DIR, 'cond_C_k10.csv')
    if not os.path.exists(cond_a_log_path):
        cond_a_log_path = os.path.join(DATA_DIR, 'cond_a.csv')
    gold_log = load_eval_log(gold_log_path, 'Gold Standard')
    cond_a_log = load_eval_log(cond_a_log_path, 'Condition A')
    cond_b_log = load_eval_log(cond_b_log_path, 'Condition B')
    cond_c_log = load_eval_log(cond_c_log_path, 'Condition C')
    logs_dict = {'Gold Standard (no latency)': gold_log, 'Condition A (k=0, padded, curriculum)': cond_a_log, 'Condition B (k=3, history, curriculum)': cond_b_log, 'Condition C (k=10, history, curriculum)': cond_c_log}
    rob_df = load_robustness(os.path.join(DATA_DIR, 'robustness_results.csv'))
    gold_zero_lat = rob_df[(rob_df['condition'].str.contains('Gold')) & (rob_df['latency'] == 0)]['mean_return'].values[0]
    print("\nGold Standard zero-latency return (reference): " + str(round(gold_zero_lat, 2)))
    print("\n--- Final Training Eval Returns ---")
    for label, df in logs_dict.items():
        final_return = df['mean_reward'].iloc[-1]
        print(label + ": " + str(round(final_return, 2)))
    lc_path = os.path.join(DATA_DIR, 'learning_curves_1_' + timestamp + '.png')
    plot_learning_curves(logs_dict, curriculum_onset_step=50000, save_path=lc_path)
    print("\nLearning curves saved to " + lc_path)
    rp_path = os.path.join(DATA_DIR, 'robustness_profile_2_' + timestamp + '.png')
    plot_robustness_profile(rob_df, gold_zero_latency_return=gold_zero_lat, save_path=rp_path)
    print("Robustness profile saved to " + rp_path)
    print("\n--- Full Robustness Table ---")
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 200)
    pivot = rob_df.pivot_table(index='condition', columns='latency', values=['mean_return', 'std_return'])
    print(pivot.to_string())
# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from collections import deque
import os

SEED = 42
DATA_DIR = 'data/'

class ActuatorLatencyWrapper(gym.Wrapper):
    def __init__(self, env, max_delay=5):
        super().__init__(env)
        self.max_delay = max_delay
        self.current_delay = 0
        self.action_dim = env.action_space.shape[0]
        self.action_buffer = deque(maxlen=max_delay + 1)

    def set_delay(self, delay):
        self.current_delay = int(np.clip(delay, 0, self.max_delay))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.action_buffer.clear()
        for _ in range(self.max_delay + 1):
            self.action_buffer.append(np.zeros(self.action_dim, dtype=np.float32))
        return obs, info

    def step(self, action):
        self.action_buffer.append(action.copy())
        buffer_list = list(self.action_buffer)
        delay_idx = max(0, len(buffer_list) - 1 - self.current_delay)
        delayed_action = buffer_list[delay_idx]
        obs, reward, terminated, truncated, info = self.env.step(delayed_action)
        if 'x_velocity' not in info:
            info['x_velocity'] = 0.0
        return obs, reward, terminated, truncated, info

def run_robustness_evaluation():
    env_name = 'HalfCheetah-v4'
    latencies = [0, 1, 2, 3, 5]
    results = {}
    for lat in latencies:
        env = ActuatorLatencyWrapper(gym.make(env_name))
        env.set_delay(lat)
        total_returns = []
        total_velocities = []
        for _ in range(10):
            obs, _ = env.reset()
            ep_return = 0
            ep_vels = []
            done = False
            while not done:
                action = np.zeros(env.action_space.shape)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += reward
                ep_vels.append(info.get('x_velocity', 0.0))
                done = terminated or truncated
            total_returns.append(ep_return)
            total_velocities.append(np.mean(ep_vels))
        results[lat] = {'return': np.mean(total_returns), 'velocity': np.mean(total_velocities)}
    print('Latency | Mean Return | Mean Velocity')
    for lat in latencies:
        print(str(lat) + ' | ' + str(results[lat]['return']) + ' | ' + str(results[lat]['velocity']))
    np.save(os.path.join(DATA_DIR, 'robustness_results.npy'), results)

if __name__ == '__main__':
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    run_robustness_evaluation()
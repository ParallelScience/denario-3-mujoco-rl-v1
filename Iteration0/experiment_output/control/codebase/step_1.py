# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import os

SEED = 42
DATA_DIR = "data/"
TOTAL_TIMESTEPS = 100_000
CURRICULUM_SWITCH = 50_000
MAX_DELAY = 3
K_B = 3
K_C = 10

class ActuatorLatencyWrapper(gym.Wrapper):
    def __init__(self, env, initial_delay=1, max_delay=5):
        super().__init__(env)
        self.current_delay = initial_delay
        self.max_delay = max_delay
        self._action_dim = env.action_space.shape[0]
        self._buffer_size = max_delay + 1
        self._action_buffer = np.zeros((self._buffer_size, self._action_dim), dtype=np.float32)
        self._buf_idx = 0
    def set_delay(self, n):
        self.current_delay = int(np.clip(n, 0, self.max_delay))
    def reset(self, **kwargs):
        self._action_buffer[:] = 0.0
        self._buf_idx = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info
    def step(self, action):
        self._action_buffer[self._buf_idx] = action
        if self.current_delay == 0:
            applied_action = action
        else:
            delayed_idx = (self._buf_idx - self.current_delay) % self._buffer_size
            applied_action = self._action_buffer[delayed_idx]
        self._buf_idx = (self._buf_idx + 1) % self._buffer_size
        obs, reward, terminated, truncated, info = self.env.step(applied_action)
        return obs, reward, terminated, truncated, info

class ActionHistoryWrapper(ActuatorLatencyWrapper):
    def __init__(self, env, k=3, initial_delay=1, max_delay=5):
        super().__init__(env, initial_delay=initial_delay, max_delay=max_delay)
        self.k = k
        self._history_buffer = np.zeros((k, self._action_dim), dtype=np.float32)
        self._history_idx = 0
        base_obs_shape = env.observation_space.shape[0]
        new_obs_dim = base_obs_shape + k * self._action_dim
        low = np.concatenate([env.observation_space.low, np.full(k * self._action_dim, -np.inf, dtype=np.float32)])
        high = np.concatenate([env.observation_space.high, np.full(k * self._action_dim, np.inf, dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    def _get_augmented_obs(self, obs):
        ordered = np.concatenate([self._history_buffer[(self._history_idx + i) % self.k] for i in range(self.k)], axis=0)
        return np.concatenate([obs, ordered], axis=0).astype(np.float32)
    def reset(self, **kwargs):
        self._history_buffer[:] = 0.0
        self._history_idx = 0
        obs, info = super().reset(**kwargs)
        return self._get_augmented_obs(obs), info
    def step(self, action):
        self._history_buffer[self._history_idx] = action
        self._history_idx = (self._history_idx + 1) % self.k
        obs, reward, terminated, truncated, info = super().step(action)
        return self._get_augmented_obs(obs), reward, terminated, truncated, info

class CurriculumLatencyCallback(BaseCallback):
    def __init__(self, curriculum_switch=50_000, max_stochastic_delay=3, verbose=0):
        super().__init__(verbose)
        self.curriculum_switch = curriculum_switch
        self.max_stochastic_delay = max_stochastic_delay
        self._latency_wrapper = None
    def _find_latency_wrapper(self, env):
        if isinstance(env, ActuatorLatencyWrapper):
            return env
        if hasattr(env, 'env'):
            return self._find_latency_wrapper(env.env)
        return None
    def _init_callback(self):
        vec_env = self.model.get_env()
        inner_env = vec_env.envs[0]
        self._latency_wrapper = self._find_latency_wrapper(inner_env)
    def _on_step(self):
        if self._latency_wrapper is None:
            return True
        dones = self.locals.get('dones', [False])
        if any(dones):
            if self.num_timesteps < self.curriculum_switch:
                new_delay = 1
            else:
                new_delay = int(np.random.randint(0, self.max_stochastic_delay + 1))
            self._latency_wrapper.set_delay(new_delay)
        return True

def make_env_A(seed):
    base_env = gym.make('HalfCheetah-v4')
    base_env.reset(seed=seed)
    latency_env = ActuatorLatencyWrapper(base_env, initial_delay=1, max_delay=MAX_DELAY)
    return Monitor(latency_env)

def make_env_B(seed, k=K_B):
    base_env = gym.make('HalfCheetah-v4')
    base_env.reset(seed=seed)
    history_env = ActionHistoryWrapper(base_env, k=k, initial_delay=1, max_delay=MAX_DELAY)
    return Monitor(history_env)

def make_env_C(seed, k=K_C):
    base_env = gym.make('HalfCheetah-v4')
    base_env.reset(seed=seed)
    history_env = ActionHistoryWrapper(base_env, k=k, initial_delay=1, max_delay=MAX_DELAY)
    return Monitor(history_env)

def extract_monitor_data(monitor_env):
    returns = np.array(monitor_env.get_episode_rewards())
    lengths = np.array(monitor_env.get_episode_lengths())
    return returns, lengths

def compute_cumulative_timesteps(episode_lengths):
    return np.cumsum(episode_lengths)

if __name__ == '__main__':
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print("Random seed used: " + str(SEED))
    env_A = make_env_A(SEED)
    env_B = make_env_B(SEED, k=K_B)
    env_C = make_env_C(SEED, k=K_C)
    obs_shape_A = env_A.observation_space.shape
    obs_shape_B = env_B.observation_space.shape
    obs_shape_C = env_C.observation_space.shape
    print("Observation space shapes:")
    print("  Condition A (Baseline, latency-only):        " + str(obs_shape_A))
    print("  Condition B (History-MLP k=3):               " + str(obs_shape_B))
    print("  Condition C (Extended-History k=10):         " + str(obs_shape_C))
    print("\n--- Training Condition A (Baseline) ---")
    model_A = SAC("MlpPolicy", env_A, seed=SEED, device='cuda', verbose=0, learning_starts=1000, batch_size=256, buffer_size=100_000, policy_kwargs=dict(net_arch=[256, 256]))
    callback_A = CurriculumLatencyCallback(curriculum_switch=CURRICULUM_SWITCH, max_stochastic_delay=MAX_DELAY)
    model_A.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_A)
    model_A.save(os.path.join(DATA_DIR, "sac_condition_A"))
    returns_A, lengths_A = extract_monitor_data(env_A)
    timesteps_A = compute_cumulative_timesteps(lengths_A)
    df_A = pd.DataFrame({'timestep': timesteps_A, 'episode_return': returns_A, 'episode_length': lengths_A})
    df_A.to_csv(os.path.join(DATA_DIR, "training_curves_A.csv"), index=False)
    final_mean_A = np.mean(returns_A[-10:]) if len(returns_A) >= 10 else np.mean(returns_A)
    print("Condition A: total episodes = " + str(len(returns_A)) + ", final mean return (last 10 eps) = " + str(round(float(final_mean_A), 2)))
    print("\n--- Training Condition B (History-MLP k=3) ---")
    model_B = SAC("MlpPolicy", env_B, seed=SEED, device='cuda', verbose=0, learning_starts=1000, batch_size=256, buffer_size=100_000, policy_kwargs=dict(net_arch=[256, 256]))
    callback_B = CurriculumLatencyCallback(curriculum_switch=CURRICULUM_SWITCH, max_stochastic_delay=MAX_DELAY)
    model_B.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_B)
    model_B.save(os.path.join(DATA_DIR, "sac_condition_B"))
    returns_B, lengths_B = extract_monitor_data(env_B)
    timesteps_B = compute_cumulative_timesteps(lengths_B)
    df_B = pd.DataFrame({'timestep': timesteps_B, 'episode_return': returns_B, 'episode_length': lengths_B})
    df_B.to_csv(os.path.join(DATA_DIR, "training_curves_B.csv"), index=False)
    final_mean_B = np.mean(returns_B[-10:]) if len(returns_B) >= 10 else np.mean(returns_B)
    print("Condition B: total episodes = " + str(len(returns_B)) + ", final mean return (last 10 eps) = " + str(round(float(final_mean_B), 2)))
    print("\n--- Training Condition C (Extended-History k=10) ---")
    model_C = SAC("MlpPolicy", env_C, seed=SEED, device='cuda', verbose=0, learning_starts=1000, batch_size=256, buffer_size=100_000, policy_kwargs=dict(net_arch=[256, 256]))
    callback_C = CurriculumLatencyCallback(curriculum_switch=CURRICULUM_SWITCH, max_stochastic_delay=MAX_DELAY)
    model_C.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_C)
    model_C.save(os.path.join(DATA_DIR, "sac_condition_C"))
    returns_C, lengths_C = extract_monitor_data(env_C)
    timesteps_C = compute_cumulative_timesteps(lengths_C)
    df_C = pd.DataFrame({'timestep': timesteps_C, 'episode_return': returns_C, 'episode_length': lengths_C})
    df_C.to_csv(os.path.join(DATA_DIR, "training_curves_C.csv"), index=False)
    final_mean_C = np.mean(returns_C[-10:]) if len(returns_C) >= 10 else np.mean(returns_C)
    print("Condition C: total episodes = " + str(len(returns_C)) + ", final mean return (last 10 eps) = " + str(round(float(final_mean_C), 2)))
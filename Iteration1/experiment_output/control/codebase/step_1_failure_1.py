# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import os
import warnings
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

warnings.filterwarnings("ignore")

STATE_DIM = 17
ACTION_DIM = 6
K_MAX = 10
TOTAL_OBS_DIM = STATE_DIM + K_MAX * ACTION_DIM

class LatencyWrapper(gym.Wrapper):
    def __init__(self, env, max_delay=10):
        super().__init__(env)
        self.max_delay = max_delay
        self.delay = 0
        self._action_buffer = deque([np.zeros(env.action_space.shape[0]) for _ in range(max_delay + 1)], maxlen=max_delay + 1)
    def set_delay(self, n):
        self.delay = int(np.clip(n, 0, self.max_delay))
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._action_buffer = deque([np.zeros(self.env.action_space.shape[0]) for _ in range(self.max_delay + 1)], maxlen=self.max_delay + 1)
        return obs, info
    def step(self, action):
        self._action_buffer.append(np.array(action, dtype=np.float32))
        if self.delay == 0:
            applied_action = action
        else:
            buffer_list = list(self._action_buffer)
            idx = max(0, len(buffer_list) - 1 - self.delay)
            applied_action = buffer_list[idx]
        return self.env.step(applied_action)

class ObservationAugmenter(gym.ObservationWrapper):
    def __init__(self, env, k=0):
        super().__init__(env)
        self.k = k
        self._action_history = deque([np.zeros(ACTION_DIM) for _ in range(k)] if k > 0 else [], maxlen=k if k > 0 else 1)
        low = np.full(TOTAL_OBS_DIM, -np.inf, dtype=np.float32)
        high = np.full(TOTAL_OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.k > 0:
            self._action_history = deque([np.zeros(ACTION_DIM) for _ in range(self.k)], maxlen=self.k)
        return self.observation(obs), info
    def observation(self, obs):
        if self.k > 0:
            action_part = np.concatenate(list(self._action_history), axis=0)
        else:
            action_part = np.zeros(0, dtype=np.float32)
        pad_size = K_MAX * ACTION_DIM - len(action_part)
        padded = np.concatenate([action_part, np.zeros(pad_size, dtype=np.float32)])
        return np.concatenate([obs.astype(np.float32), padded.astype(np.float32)])
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.k > 0:
            self._action_history.append(np.array(action, dtype=np.float32))
        return self.observation(obs), reward, terminated, truncated, info

class CurriculumLatencyCallback(BaseCallback):
    def __init__(self, latency_env, warmup_steps=50000, ramp_end_steps=500000, max_delay=5, verbose=0):
        super().__init__(verbose=verbose)
        self.latency_env = latency_env
        self.warmup_steps = warmup_steps
        self.ramp_end_steps = ramp_end_steps
        self.max_delay = max_delay
    def _get_current_max_delay(self):
        n = self.num_timesteps
        if n <= self.warmup_steps:
            return 0.0
        ramp_steps = self.ramp_end_steps - self.warmup_steps
        progress = min(1.0, (n - self.warmup_steps) / ramp_steps)
        return progress * self.max_delay
    def _on_step(self):
        dones = self.locals.get("dones", None)
        if dones is not None and np.any(dones):
            current_max = self._get_current_max_delay()
            new_delay = int(np.floor(np.random.uniform(0, current_max + 1))) if current_max > 0 else 0
            self.latency_env.set_delay(min(new_delay, self.max_delay))
        return True

if __name__ == '__main__':
    np.random.seed(42)
    data_dir = "data/"
    TOTAL_STEPS = 500000
    EVAL_FREQ = 20000
    for k in [0, 3, 10]:
        env = Monitor(ObservationAugmenter(LatencyWrapper(gym.make("HalfCheetah-v4")), k=k))
        eval_env = Monitor(ObservationAugmenter(LatencyWrapper(gym.make("HalfCheetah-v4")), k=k))
        model = SAC("MlpPolicy", env, verbose=0)
        cb = CurriculumLatencyCallback(env.env.env)
        eval_cb = EvalCallback(eval_env, eval_freq=EVAL_FREQ, log_path=data_dir, best_model_save_path=data_dir)
        model.learn(total_timesteps=TOTAL_STEPS, callback=[cb, eval_cb])
        model.save(os.path.join(data_dir, "model_k" + str(k)))
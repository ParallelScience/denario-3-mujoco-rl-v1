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
warnings.filterwarnings('ignore')
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
STATE_DIM = 17
ACTION_DIM = 6
K_MAX = 10
TOTAL_OBS_DIM = STATE_DIM + K_MAX * ACTION_DIM
class LatencyWrapper(gym.Wrapper):
    def __init__(self, env, max_delay=10):
        super().__init__(env)
        self.max_delay = max_delay
        self.delay = 0
        self._buf = deque([np.zeros(env.action_space.shape[0], dtype=np.float32) for _ in range(max_delay + 1)], maxlen=max_delay + 1)
    def set_delay(self, n):
        self.delay = int(np.clip(n, 0, self.max_delay))
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._buf = deque([np.zeros(self.env.action_space.shape[0], dtype=np.float32) for _ in range(self.max_delay + 1)], maxlen=self.max_delay + 1)
        return obs, info
    def step(self, action):
        self._buf.append(np.array(action, dtype=np.float32))
        if self.delay == 0:
            applied = action
        else:
            buf_list = list(self._buf)
            idx = max(0, len(buf_list) - 1 - self.delay)
            applied = buf_list[idx]
        return self.env.step(applied)
class ObservationAugmenter(gym.ObservationWrapper):
    def __init__(self, env, k=0):
        super().__init__(env)
        self.k = k
        self._hist = deque([np.zeros(ACTION_DIM, dtype=np.float32) for _ in range(max(k, 1))], maxlen=max(k, 1))
        low = np.full(TOTAL_OBS_DIM, -np.inf, dtype=np.float32)
        high = np.full(TOTAL_OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._hist = deque([np.zeros(ACTION_DIM, dtype=np.float32) for _ in range(max(self.k, 1))], maxlen=max(self.k, 1))
        return self.observation(obs), info
    def observation(self, obs):
        if self.k > 0:
            action_part = np.concatenate(list(self._hist), axis=0)
        else:
            action_part = np.zeros(0, dtype=np.float32)
        augmented = np.zeros(TOTAL_OBS_DIM, dtype=np.float32)
        augmented[:STATE_DIM] = obs.astype(np.float32)
        if self.k > 0:
            augmented[STATE_DIM:STATE_DIM + len(action_part)] = action_part
        return augmented
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.k > 0:
            self._hist.append(np.array(action, dtype=np.float32))
        return self.observation(obs), reward, terminated, truncated, info
class CurriculumLatencyCallback(BaseCallback):
    def __init__(self, latency_env, warmup_steps=50000, ramp_end_steps=300000, max_delay=5, verbose=0):
        super().__init__(verbose=verbose)
        self.latency_env = latency_env
        self.warmup_steps = warmup_steps
        self.ramp_end_steps = ramp_end_steps
        self.max_delay = max_delay
    def _current_max_delay(self):
        n = self.num_timesteps
        if n <= self.warmup_steps: return 0.0
        progress = min(1.0, (n - self.warmup_steps) / (self.ramp_end_steps - self.warmup_steps))
        return progress * self.max_delay
    def _on_step(self):
        dones = self.locals.get('dones', None)
        if dones is not None and np.any(dones):
            cur_max = self._current_max_delay()
            new_delay = int(np.floor(np.random.uniform(0.0, cur_max + 1.0))) if cur_max > 0.0 else 0
            self.latency_env.set_delay(min(new_delay, self.max_delay))
        return True
def build_train_env(k):
    base = gym.make('HalfCheetah-v4')
    lat = LatencyWrapper(base, max_delay=10)
    aug = ObservationAugmenter(lat, k=k)
    return Monitor(aug), lat
def build_eval_env(k):
    base = gym.make('HalfCheetah-v4')
    lat = LatencyWrapper(base, max_delay=10)
    lat.set_delay(0)
    aug = ObservationAugmenter(lat, k=k)
    return Monitor(aug)
if __name__ == '__main__':
    np.random.seed(42)
    data_dir = 'data/'
    os.makedirs(data_dir, exist_ok=True)
    TOTAL_STEPS = 300000
    EVAL_FREQ = 20000
    policy_kwargs = dict(net_arch=[256, 256])
    for k in [0, 3, 10]:
        train_env, lat_ref = build_train_env(k)
        eval_env = build_eval_env(k)
        log_dir = os.path.join(data_dir, 'k' + str(k) + '_log')
        os.makedirs(log_dir, exist_ok=True)
        cb = CurriculumLatencyCallback(lat_ref, ramp_end_steps=TOTAL_STEPS)
        eval_cb = EvalCallback(eval_env, log_path=log_dir, eval_freq=EVAL_FREQ, deterministic=True)
        model = SAC('MlpPolicy', train_env, policy_kwargs=policy_kwargs, batch_size=512, learning_starts=10000, gradient_steps=2, device='cuda', verbose=0)
        model.learn(total_timesteps=TOTAL_STEPS, callback=[cb, eval_cb])
        model.save(os.path.join(data_dir, 'model_k' + str(k)))
        print('Finished training k=' + str(k))
    train_env = Monitor(gym.make('HalfCheetah-v4'))
    eval_env = Monitor(gym.make('HalfCheetah-v4'))
    log_dir = os.path.join(data_dir, 'gold_log')
    os.makedirs(log_dir, exist_ok=True)
    eval_cb = EvalCallback(eval_env, log_path=log_dir, eval_freq=EVAL_FREQ, deterministic=True)
    model = SAC('MlpPolicy', train_env, policy_kwargs=policy_kwargs, batch_size=512, learning_starts=10000, gradient_steps=2, device='cuda', verbose=0)
    model.learn(total_timesteps=TOTAL_STEPS, callback=[eval_cb])
    model.save(os.path.join(data_dir, 'gold_model'))
    print('Finished training Gold Standard')
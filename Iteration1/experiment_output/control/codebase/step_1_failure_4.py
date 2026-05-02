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
import warnings
warnings.filterwarnings('ignore')
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
STATE_DIM = 17
ACTION_DIM = 6
K_MAX = 10
TOTAL_OBS_DIM = STATE_DIM + K_MAX * ACTION_DIM
class LatencyWrapper(gym.Wrapper):
    def __init__(self, env, max_delay=10):
        super().__init__(env)
        self.max_delay = max_delay
        self.delay = 0
        self._act_dim = env.action_space.shape[0]
        self._buf = np.zeros((max_delay + 1, self._act_dim), dtype=np.float32)
        self._ptr = 0
    def set_delay(self, n):
        self.delay = int(np.clip(n, 0, self.max_delay))
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._buf[:] = 0.0
        self._ptr = 0
        return obs, info
    def step(self, action):
        self._buf[self._ptr] = action
        if self.delay == 0:
            applied = action
        else:
            read_ptr = (self._ptr - self.delay) % (self.max_delay + 1)
            applied = self._buf[read_ptr]
        self._ptr = (self._ptr + 1) % (self.max_delay + 1)
        return self.env.step(applied)
class ObservationAugmenter(gym.ObservationWrapper):
    def __init__(self, env, k=0):
        super().__init__(env)
        self.k = k
        self._aug = np.zeros(TOTAL_OBS_DIM, dtype=np.float32)
        if k > 0:
            self._hist_buf = np.zeros((k, ACTION_DIM), dtype=np.float32)
            self._hist_ptr = 0
        low = np.full(TOTAL_OBS_DIM, -np.inf, dtype=np.float32)
        high = np.full(TOTAL_OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.k > 0:
            self._hist_buf[:] = 0.0
            self._hist_ptr = 0
        return self.observation(obs), info
    def observation(self, obs):
        self._aug[:STATE_DIM] = obs
        if self.k > 0:
            ordered = np.roll(self._hist_buf, -self._hist_ptr, axis=0)
            self._aug[STATE_DIM:STATE_DIM + self.k * ACTION_DIM] = ordered.ravel()
            self._aug[STATE_DIM + self.k * ACTION_DIM:] = 0.0
        else:
            self._aug[STATE_DIM:] = 0.0
        return self._aug.copy()
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.k > 0:
            self._hist_buf[self._hist_ptr] = action
            self._hist_ptr = (self._hist_ptr + 1) % self.k
        return self.observation(obs), reward, terminated, truncated, info
class CurriculumLatencyCallback(BaseCallback):
    def __init__(self, latency_envs, warmup_steps=20000, ramp_end_steps=100000, max_delay=5, verbose=0):
        super().__init__(verbose=verbose)
        self.latency_envs = latency_envs
        self.warmup_steps = warmup_steps
        self.ramp_end_steps = ramp_end_steps
        self.max_delay = max_delay
    def _current_max_delay(self):
        n = self.num_timesteps
        if n <= self.warmup_steps: return 0.0
        progress = min(1.0, (n - self.warmup_steps) / float(self.ramp_end_steps - self.warmup_steps))
        return progress * self.max_delay
    def _on_step(self):
        dones = self.locals.get('dones', None)
        if dones is not None:
            cur_max = self._current_max_delay()
            for i, done in enumerate(dones):
                if done:
                    new_delay = int(np.floor(np.random.uniform(0.0, cur_max + 1.0))) if cur_max > 0.0 else 0
                    self.latency_envs[i].set_delay(min(new_delay, self.max_delay))
        return True
def make_single_train_env(k):
    def _init():
        base = gym.make('HalfCheetah-v4')
        lat = LatencyWrapper(base, max_delay=10)
        aug = ObservationAugmenter(lat, k=k)
        return Monitor(aug)
    return _init
def build_vec_train_env(k, n_envs=4):
    fns = [make_single_train_env(k) for _ in range(n_envs)]
    vec_env = SubprocVecEnv(fns)
    return VecMonitor(vec_env)
def build_eval_env(k):
    base = gym.make('HalfCheetah-v4')
    lat = LatencyWrapper(base, max_delay=10)
    lat.set_delay(0)
    aug = ObservationAugmenter(lat, k=k)
    return Monitor(aug)
def save_eval_log_csv(log_dir, csv_path):
    npz_path = os.path.join(log_dir, 'evaluations.npz')
    data = np.load(npz_path)
    timesteps = data['timesteps']
    mean_rewards = data['results'].mean(axis=1)
    df = pd.DataFrame({'timestep': timesteps, 'mean_reward': mean_rewards})
    df.to_csv(csv_path, index=False)
    return timesteps, mean_rewards
if __name__ == '__main__':
    np.random.seed(42)
    data_dir = 'data/'
    os.makedirs(data_dir, exist_ok=True)
    TOTAL_STEPS = 100000
    EVAL_FREQ = 20000
    N_EVAL_EPS = 3
    WARMUP = 20000
    N_ENVS = 4
    sac_kwargs = dict(policy_kwargs=dict(net_arch=[256, 256]), learning_rate=3e-4, buffer_size=200000, batch_size=512, learning_starts=5000, train_freq=1, gradient_steps=1, device='cuda', verbose=0, seed=42)
    gold_train_env = make_vec_env('HalfCheetah-v4', n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    gold_eval_env = Monitor(gym.make('HalfCheetah-v4'))
    gold_log_dir = os.path.join(data_dir, 'gold_log')
    os.makedirs(gold_log_dir, exist_ok=True)
    gold_eval_cb = EvalCallback(gold_eval_env, log_path=gold_log_dir, eval_freq=max(EVAL_FREQ // N_ENVS, 1), n_eval_episodes=N_EVAL_EPS, deterministic=True, verbose=0)
    gold_model = SAC('MlpPolicy', gold_train_env, **sac_kwargs)
    gold_model.learn(total_timesteps=TOTAL_STEPS, callback=[gold_eval_cb])
    gold_model.save(os.path.join(data_dir, 'gold_model'))
    gold_train_env.close()
    gold_eval_env.close()
    save_eval_log_csv(gold_log_dir, os.path.join(data_dir, 'gold_eval_log.csv'))
    for cname, k in zip(['cond_A_k0', 'cond_B_k3', 'cond_C_k10'], [0, 3, 10]):
        train_env = build_vec_train_env(k, n_envs=N_ENVS)
        eval_env = build_eval_env(k)
        log_dir = os.path.join(data_dir, cname + '_log')
        os.makedirs(log_dir, exist_ok=True)
        lat_refs = [env.unwrapped for env in train_env.envs]
        curriculum_cb = CurriculumLatencyCallback(lat_refs, warmup_steps=WARMUP, ramp_end_steps=TOTAL_STEPS, max_delay=5, verbose=0)
        eval_cb = EvalCallback(eval_env, log_path=log_dir, eval_freq=max(EVAL_FREQ // N_ENVS, 1), n_eval_episodes=N_EVAL_EPS, deterministic=True, verbose=0)
        model = SAC('MlpPolicy', train_env, **sac_kwargs)
        model.learn(total_timesteps=TOTAL_STEPS, callback=[curriculum_cb, eval_cb])
        model.save(os.path.join(data_dir, cname + '_model'))
        train_env.close()
        eval_env.close()
        save_eval_log_csv(log_dir, os.path.join(data_dir, cname + '_eval_log.csv'))
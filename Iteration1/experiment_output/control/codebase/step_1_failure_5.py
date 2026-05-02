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
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

DATA_DIR = "data/"
STATE_DIM = 17
ACTION_DIM = 6
K_MAX = 10
TOTAL_OBS_DIM = STATE_DIM + K_MAX * ACTION_DIM
TOTAL_STEPS = 500_000
EVAL_FREQ = 20_000
CURRICULUM_START = 50_000
MAX_DELAY = 5
SEED = 42

class LatencyWrapper(gym.Wrapper):
    def __init__(self, env, initial_delay=0):
        super().__init__(env)
        self._delay = initial_delay
        self._action_dim = env.action_space.shape[0]
        self._buffer = deque([np.zeros(self._action_dim, dtype=np.float32)] * (MAX_DELAY + 1), maxlen=MAX_DELAY + 1)
    def set_delay(self, n):
        self._delay = int(n)
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._buffer = deque([np.zeros(self._action_dim, dtype=np.float32)] * (MAX_DELAY + 1), maxlen=MAX_DELAY + 1)
        return obs, info
    def step(self, action):
        self._buffer.append(np.array(action, dtype=np.float32))
        delayed_action = self._buffer[-(self._delay + 1)]
        return self.env.step(delayed_action)

class ObservationAugmenter(gym.ObservationWrapper):
    def __init__(self, env, k=0):
        super().__init__(env)
        assert 0 <= k <= K_MAX
        self._k = k
        self._action_dim = env.action_space.shape[0]
        self._action_history = deque([np.zeros(self._action_dim, dtype=np.float32)] * k, maxlen=max(k, 1))
        low = np.full(TOTAL_OBS_DIM, -np.inf, dtype=np.float32)
        high = np.full(TOTAL_OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._k > 0:
            self._action_history = deque([np.zeros(self._action_dim, dtype=np.float32)] * self._k, maxlen=self._k)
        return self.observation(obs), info
    def observation(self, obs):
        history = np.concatenate(list(self._action_history), axis=0) if self._k > 0 else np.array([], dtype=np.float32)
        pad_len = K_MAX * ACTION_DIM - len(history)
        padded = np.concatenate([history, np.zeros(pad_len, dtype=np.float32)])
        return np.concatenate([obs.astype(np.float32), padded]).astype(np.float32)
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._k > 0:
            self._action_history.append(np.array(action, dtype=np.float32))
        return self.observation(obs), reward, terminated, truncated, info

class CurriculumLatencyCallback(BaseCallback):
    def __init__(self, latency_env, verbose=0):
        super().__init__(verbose=verbose)
        self._latency_env = latency_env
    def _on_step(self):
        dones = self.locals.get("dones", [False])
        if any(dones):
            self._update_delay()
        return True
    def _on_training_start(self):
        self._latency_env.set_delay(0)
    def _update_delay(self):
        n = self.num_timesteps
        current_max = 0.0 if n <= CURRICULUM_START else min((n - CURRICULUM_START) / (TOTAL_STEPS - CURRICULUM_START), 1.0) * MAX_DELAY
        new_delay = int(np.floor(np.random.uniform(0, current_max + 1e-9))) if current_max >= 1e-6 else 0
        self._latency_env.set_delay(min(new_delay, MAX_DELAY))

def load_eval_log(log_dir):
    data = np.load(os.path.join(log_dir, "evaluations.npz"))
    return pd.DataFrame({"timesteps": data["timesteps"], "mean_reward": data["results"].mean(axis=1)})

def train_condition(k, label, csv_name, model_name):
    latency_env = LatencyWrapper(gym.make("HalfCheetah-v4"), initial_delay=0)
    train_vec = DummyVecEnv([lambda: Monitor(ObservationAugmenter(latency_env, k=k))])
    eval_vec = DummyVecEnv([lambda: Monitor(ObservationAugmenter(LatencyWrapper(gym.make("HalfCheetah-v4")), k=k))])
    log_dir = os.path.join(DATA_DIR, model_name + "_logs")
    eval_cb = EvalCallback(eval_vec, log_path=log_dir, eval_freq=EVAL_FREQ, n_eval_episodes=10, deterministic=True, verbose=0)
    model = SAC("MlpPolicy", train_vec, verbose=0, device="cuda", seed=SEED)
    model.learn(total_timesteps=TOTAL_STEPS, callback=[eval_cb, CurriculumLatencyCallback(latency_env)], progress_bar=False)
    eval_log = load_eval_log(log_dir)
    model.save(os.path.join(DATA_DIR, model_name))
    eval_log.to_csv(os.path.join(DATA_DIR, csv_name), index=False)
    print(label + " final eval return: " + str(round(eval_log["mean_reward"].iloc[-1], 2)))

if __name__ == '__main__':
    assert gym.make("HalfCheetah-v4").observation_space.shape[0] == 17
    assert gym.make("HalfCheetah-v4").action_space.shape[0] == 6
    train_condition(0, "Condition A", "cond_a.csv", "cond_a_model")
    train_condition(3, "Condition B", "cond_b.csv", "cond_b_model")
    train_condition(10, "Condition C", "cond_c.csv", "cond_c_model")
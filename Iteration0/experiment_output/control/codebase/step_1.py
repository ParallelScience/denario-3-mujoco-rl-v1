# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from collections import deque
import random
import os
import time

SEED = 42
TOTAL_STEPS = 500000
EVAL_INTERVAL = 10000
EVAL_EPISODES = 5
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 1000000
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005
HIDDEN_DIM = 256
GRU_HIDDEN_DIM = 256
SEQ_LEN = 20
K_HISTORY = 3
CURRICULUM_START = 100000
GRAD_CLIP_MLP = 1.0
GRAD_CLIP_GRU = 0.5
WARMUP_STEPS = 10000
DATA_DIR = 'data/'
LOG_STD_MIN = -20
LOG_STD_MAX = 2
MAX_EPISODES_SEQ_BUF = 5000

class ActuatorLatencyWrapper(gym.Wrapper):
    def __init__(self, env, max_delay=5):
        super().__init__(env)
        self.max_delay = max_delay
        self.current_delay = 1
        self.action_dim = env.action_space.shape[0]
        self.action_buffer = deque(maxlen=max_delay + 1)
        self.last_delayed_action = np.zeros(self.action_dim, dtype=np.float32)

    def set_delay(self, delay):
        self.current_delay = int(np.clip(delay, 0, self.max_delay))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.action_buffer.clear()
        for _ in range(self.max_delay + 1):
            self.action_buffer.append(np.zeros(self.action_dim, dtype=np.float32))
        self.last_delayed_action = np.zeros(self.action_dim, dtype=np.float32)
        return obs, info

    def step(self, action):
        self.action_buffer.append(action.copy())
        buffer_list = list(self.action_buffer)
        delay_idx = max(0, len(buffer_list) - 1 - self.current_delay)
        delayed_action = buffer_list[delay_idx]
        self.last_delayed_action = delayed_action.copy()
        obs, reward, terminated, truncated, info = self.env.step(delayed_action)
        return obs, reward, terminated, truncated, info

if __name__ == '__main__':
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    print('Training pipeline initialized.')
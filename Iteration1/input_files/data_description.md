# Deep Reinforcement Learning on MuJoCo Continuous Control Benchmarks

## Overview

This project investigates novel reinforcement learning methods on standard MuJoCo continuous-control benchmarks from the Gymnasium library (gymnasium[mujoco], mujoco 3.8.0). MuJoCo provides physically realistic simulation of articulated robotic bodies, making it the standard testbed for evaluating deep RL algorithms on high-dimensional continuous action spaces.

## Available Environments

All standard MuJoCo Gymnasium environments are available:

- **HalfCheetah-v4/v5** — 17-dim state, 6-dim action. Fast locomotion task for a 2D cheetah-like body. No termination condition. Dense reward based on forward velocity.
- **Ant-v4/v5** — 27-dim state, 8-dim action. Ant locomotion in 3D. Episodes terminate if the ant falls over. Dense reward based on forward speed and control cost.
- **Hopper-v4/v5** — 11-dim state, 3-dim action. Single-legged hopper. Terminates on falling. Requires balance and coordination.
- **Walker2d-v4/v5** — 17-dim state, 6-dim action. Bipedal walker. Terminates on falling. Requires balance and forward locomotion.
- **Humanoid-v4/v5** — 376-dim state, 17-dim action. Full humanoid body. High-dimensional, challenging exploration.
- **Swimmer-v4/v5** — 8-dim state, 2-dim action. Underwater swimmer. Simple dynamics, no termination.
- **Reacher-v4/v5** — 11-dim state, 2-dim action. 2-DOF arm reaching task. Sparse-ish reward.
- **InvertedPendulum-v4/v5** — simple control baseline.

## Hardware

- NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM), CUDA 13.0
- PyTorch available at device='cuda'
- 32 CPUs, 64 GB RAM
- Parallel environment vectorization supported via gymnasium.vector

## Standard Baselines

For reference, well-known algorithms and their typical performance on these environments:
- **SAC** (Soft Actor-Critic): state-of-the-art off-policy, HalfCheetah ~12,000 return, Ant ~5,000, Hopper ~3,500
- **PPO**: on-policy, slightly lower sample efficiency
- **TD3**: deterministic off-policy, strong on HalfCheetah
- **DDPG**: original off-policy actor-critic

## Research Directions of Interest

Open questions in continuous-control RL include:
- Sample efficiency improvements
- Transfer learning and generalization across tasks
- Incorporating physical priors or structure into policy/value networks
- Reward shaping and curriculum learning
- Multi-task and meta-learning
- Safe RL / constraint satisfaction
- Exploration strategies in high-dimensional spaces

## File Paths

No pre-existing data files. All experience is generated online via environment interaction.
Environments instantiated as: `gymnasium.make('HalfCheetah-v4')` etc.

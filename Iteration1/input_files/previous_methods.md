**CRITICAL IMPLEMENTATION REQUIREMENT**: Use `stable-baselines3` (SB3) for all SAC training. Do NOT implement SAC from scratch. Do NOT generate synthetic or simulated training curves. All results must come from real environment interaction and real model training.

1. **Environment Wrapper Implementation**: 
   - Develop a custom `gymnasium.Wrapper` that simulates actuator latency via a circular action buffer.
   - The wrapper delays the applied action by `current_delay` steps.
   - Implement a `set_delay(n)` method to allow changing the delay during training.
   - For Condition B (History-MLP): augment the observation by concatenating the last k=3 actions to the state vector.

2. **SAC Training with stable-baselines3**:
   - Use `stable_baselines3.SAC` for all conditions. Do NOT write a custom SAC.
   - **Condition A (Baseline)**: `SAC("MlpPolicy", env_A, ...)` — standard MLP, no history.
   - **Condition B (History-MLP)**: `SAC("MlpPolicy", env_B, ...)` where `env_B` is the latency wrapper with observation augmented by last k=3 actions.
   - **Condition C (GRU-SAC)**: `SAC("MlpPolicy", env_C, policy_kwargs=dict(net_arch=[256,256]))` — for simplicity in this PoC, use a longer action history (k=5 or k=10 actions concatenated) as a proxy for temporal context, since SB3 does not natively support GRU policies. Label this "Extended-History SAC" in reporting.
   - Training: **100,000 steps** per condition (not 500k — reduce for PoC speed).
   - Use `device='cuda'` for GPU training.

3. **Curriculum Latency Schedule**:
   - First 50,000 steps: fixed 1-step delay.
   - Steps 50,000–100,000: stochastic delay from Uniform(0, 3) steps, resampled each episode.
   - Implement this by updating `env.set_delay(...)` at each episode start.
   - Use SB3's `callback` mechanism (e.g., `BaseCallback`) to update the delay schedule during training.

4. **Training Execution**:
   - Train each condition sequentially using `model.learn(total_timesteps=100_000, callback=latency_callback)`.
   - Log episode rewards using SB3's built-in logging (tensorboard or `EvalCallback`).
   - Save each trained model: `model.save("data/sac_condition_A")` etc.

5. **Robustness Evaluation**:
   - After training, evaluate each saved model at fixed latencies: 0, 1, 2, 3, 5 steps.
   - For each latency: run 5 evaluation episodes using `model.predict(obs, deterministic=True)`.
   - Record mean episode return and mean forward velocity (from `info['x_velocity']`).

6. **Visualization**:
   - Plot learning curves: episode return vs training steps for all 3 conditions.
   - Plot robustness profiles: mean return vs latency level for all 3 conditions.
   - Save all plots as PNG files in the data directory.

7. **Reporting**:
   - Report only results from actual trained models evaluated in the real environment.
   - If training fails or produces poor results, report that honestly.
   - Compare final training returns and robustness profiles across the 3 conditions.

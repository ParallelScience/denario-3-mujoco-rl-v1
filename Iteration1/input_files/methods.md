1. **Gold Standard Baseline Establishment**:
   - Train a standard SAC agent on the original `HalfCheetah-v4` (zero latency) for 200,000 steps.
   - Use 5 random seeds to establish a high-performance, stable baseline.
   - Save these models to serve as the "Gold Standard" reference for calculating the "Cost of Robustness."

2. **Environment Wrapper Implementation**:
   - Implement a `LatencyWrapper` that maintains a circular buffer of the last 10 actions.
   - The wrapper must support a `set_delay(n)` method to inject a specific delay into the observation-action loop.
   - Implement an `ObservationAugmenter` wrapper that concatenates the last $k$ actions to the state. To ensure identical input dimensionality across all conditions, pad the input vector with zeros to match the size of the $k=10$ history-augmented state (17 state dims + 60 action dims = 77 total).

3. **Experimental Conditions**:
   - **Condition A (Baseline)**: Standard SAC, 0-latency training, input padded to 77 dimensions.
   - **Condition B (History-MLP, k=3)**: SAC, curriculum-latency training, 3-action history + 7 zeros (to reach 60 action-dim padding), total 77 dimensions.
   - **Condition C (History-MLP, k=10)**: SAC, curriculum-latency training, 10-action history, total 77 dimensions.
   - All conditions must use an identical MLP architecture (e.g., [256, 256]) to ensure network capacity is not a confounding variable.

4. **Curriculum Latency Schedule**:
   - Implement a linear ramp-up for the maximum delay magnitude.
   - For the first 50,000 steps, maintain 0-latency to allow the agent to learn basic locomotion.
   - From 50,000 to 200,000 steps, linearly increase the maximum delay magnitude from 0 to 5 steps.
   - Use a `BaseCallback` to sample the delay from `Uniform(0, current_max_delay)` at the start of every episode.

5. **Training Protocol**:
   - Train all conditions (A, B, and C) for 200,000 steps using 5 random seeds per condition.
   - Utilize `stable-baselines3` SAC implementation with `device='cuda'`.
   - Use `EvalCallback` to monitor performance on a zero-latency environment every 5,000 steps to track the "Cost of Robustness."

6. **Robustness Evaluation**:
   - Evaluate all saved models (from all 5 seeds) at fixed latency levels: 0, 1, 2, 3, 4, and 5 steps.
   - Use a fixed, deterministic environment seed for all evaluations to reduce variance and isolate the impact of latency.
   - Record mean cumulative return and mean forward velocity for each condition at each latency level.

7. **Statistical Analysis**:
   - Calculate the mean and 95% confidence intervals for returns across the 5 seeds for each condition.
   - Quantify the "Performance Gap" at 0-latency by comparing the latency-trained models against the Gold Standard.

8. **Visualization**:
   - Generate learning curves (mean return vs. training steps) with shaded confidence intervals for all conditions.
   - Create a "Robustness Profile" plot: mean return (y-axis) vs. latency level (x-axis) for all conditions, including the Gold Standard as a static reference line to visualize the trade-off between history length and delay tolerance.
**CRITICAL IMPLEMENTATION REQUIREMENT**: Use `stable-baselines3` SAC for all training. Do NOT implement SAC from scratch. Do NOT generate synthetic training curves. All results must come from real environment interaction.

## Three fixes from Iteration 0

### Fix 1: Valid zero-latency baseline
### Fix 2: Decoupled input dimensionality (padding)
### Fix 3: Smooth curriculum (linear ramp, no hard switch)

---

1. **Gold Standard Baseline**:
   - Train a standard SAC agent on HalfCheetah-v4 with **zero latency** for 200,000 steps (1 seed).
   - This establishes peak reachable performance without any latency, serving as the reference ceiling.

2. **Environment Wrapper**:
   - Implement `LatencyWrapper`: circular action buffer, `set_delay(n)` method to inject delay.
   - Implement `ObservationAugmenter`: concatenates last k actions to the state.
   - **Fixed input dimensionality**: pad all conditions to the same observation size (77 dims = 17 state + 10 actions × 6). Condition A: pad with zeros. Condition B (k=3): 3 real actions + 7 zero-padded. Condition C (k=10): 10 real actions. All use identical MLP [256, 256].

3. **Experimental Conditions** (3 conditions, 1 seed each for PoC, 200k steps):
   - **Condition A (Padded Baseline)**: SAC on HalfCheetah with latency curriculum, observation padded to 77 dims with zeros (no real history). Tests whether the baseline failure was due to the curriculum or the lack of history.
   - **Condition B (History k=3)**: 3-action history + zero padding → 77 dims.
   - **Condition C (History k=10)**: 10-action history → 77 dims.

4. **Smooth Curriculum Latency Schedule**:
   - Steps 0–50,000: zero latency (agent learns basic locomotion first).
   - Steps 50,000–200,000: linearly ramp maximum delay from 0 to 5 steps. At each episode start, sample delay from Uniform(0, current_max_delay).
   - Implement via SB3 `BaseCallback`.

5. **Training Protocol**:
   - Train all 4 conditions (Gold Standard + A + B + C) using `stable_baselines3.SAC` with `device='cuda'`.
   - 200,000 steps each, 1 seed.
   - Use `EvalCallback` (zero-latency eval env) every 10,000 steps to track learning progress.
   - Save each trained model.

6. **Robustness Evaluation**:
   - After training, evaluate each model at fixed latencies: 0, 1, 2, 3, 5 steps.
   - 10 evaluation episodes per latency level, deterministic policy.
   - Record mean episode return and mean forward velocity.

7. **Visualization**:
   - Learning curves: mean return vs. training steps for all 4 conditions.
   - Robustness profile: mean return vs. latency level for all conditions, with Gold Standard as a horizontal reference line.
   - Save all plots as PNG.

8. **Reporting**:
   - Compare final training returns and robustness profiles across conditions.
   - Quantify the "cost of robustness": performance gap at 0-latency vs. Gold Standard.
   - Report honestly if any condition fails to learn.

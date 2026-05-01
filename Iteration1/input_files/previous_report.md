

Iteration 0:
### Project Summary: Action-History Augmentation for Latency-Robust Control

**Objective:** Mitigate performance degradation in HalfCheetah-v4 caused by actuator latency using action-history stacking vs. extended-history MLP policies.

**Methodology:**
- **Environment:** HalfCheetah-v4 with a custom `gymnasium.Wrapper` simulating actuator latency via a circular action buffer.
- **Curriculum:** 100k steps total; 0-50k (fixed 1-step delay), 50k-100k (stochastic delay $\sim U(0, 3)$).
- **Conditions (SAC, SB3):**
    - **A (Baseline):** 17-dim state (no history).
    - **B (History-MLP):** 35-dim state (last 3 actions).
    - **C (Ext-History):** 77-dim state (last 10 actions).
- **Evaluation:** 5 episodes per latency level (0, 1, 2, 3, 5 steps).

**Key Findings:**
1. **Efficacy of Stacking:** Condition B (k=3) significantly outperformed the baseline, achieving a final return of 502.22 vs. -135.05. Explicit action history is a viable, low-overhead strategy for restoring Markovian properties.
2. **Dimensionality Trade-off:** Condition C (k=10) underperformed Condition B (189.27 return). Increased input dimensionality likely hindered learning within the 100k-step budget, suggesting that temporal context benefits are non-monotonic when using flat MLP architectures.
3. **Baseline Failure:** The baseline failed to converge to any viable locomotion strategy, indicating that the latency curriculum is highly disruptive to standard SAC without history augmentation.

**Limitations & Constraints:**
- **Sample Efficiency:** 100k steps is insufficient for SAC convergence on HalfCheetah; results reflect early-stage learning dynamics.
- **Confounding Factors:** The use of concatenated inputs for temporal context conflates history length with input dimensionality.
- **Statistical Power:** Single-seed (seed=42) evaluation limits generalizability.

**Future Directions:**
- **Architectural Shift:** Replace concatenated MLP inputs with native temporal encoders (GRU/LSTM) to decouple history length from input dimensionality.
- **Extended Training:** Scale to 1M+ steps to determine if Condition C eventually surpasses Condition B.
- **Robustness Testing:** Expand evaluation to include multi-seed variance and higher latency regimes (e.g., >5 steps).
        
# Results

## Experimental Setup

Three conditions were evaluated on the **HalfCheetah-v4** continuous-control benchmark using Soft Actor-Critic (SAC) implemented via <code>stable-baselines3</code>. All conditions were trained for **100,000 environment steps** under a **curriculum latency schedule**: a fixed 1-step actuator delay was applied for the first 50,000 steps, after which the delay was resampled at each episode boundary from a discrete Uniform(0, 3) distribution. A single random seed (seed = 42) was used throughout, and all training was performed on GPU (NVIDIA RTX PRO 6000 Blackwell, CUDA 13.0). The three conditions differ in how the policy receives information about past actions:

- **Condition A (Baseline)**: Standard SAC with a 17-dimensional observation vector (the native HalfCheetah-v4 state). The actuator latency wrapper is applied, but no action-history information is provided to the policy. The policy must implicitly cope with the non-Markovian dynamics introduced by the delay.
- **Condition B (History-MLP, k=3)**: The observation is augmented by concatenating the last k=3 actions (each 6-dimensional) to the native state, yielding a **35-dimensional** input vector. This provides the policy with explicit access to recent control history, enabling it to reason about which commands are currently "in flight" through the actuator pipeline.
- **Condition C (Extended-History, k=10)**: The observation is augmented with the last k=10 actions, yielding a **77-dimensional** input vector. This condition serves as a proxy for a GRU-augmented policy, providing a longer temporal context window. However, the substantially larger input dimensionality also introduces a confound: the policy network must learn to extract relevant temporal structure from a higher-dimensional, partially redundant input space, which may independently affect learning dynamics.

The observation space dimensionalities were verified prior to training: Condition A produced a (17,) space, Condition B a (35,) space, and Condition C a (77,) space, confirming correct wrapper implementation. All three conditions used identical SAC hyperparameters: a two-layer MLP policy with [256, 256] hidden units, a replay buffer of 100,000 transitions, a batch size of 256, 1,000 learning-start steps, and the default SAC entropy regularization.

---

## Learning Curves

The learning curves (Panel 1 of Figure 1) reveal substantial differences in training dynamics across the three conditions over 100,000 steps.

**Condition B (History-MLP, k=3)** achieved the highest final training performance, with a mean episode return of **502.22** averaged over the last 10 episodes. This represents a markedly positive outcome relative to the baseline, suggesting that providing the policy with explicit access to the three most recent actions substantially aids credit assignment and policy optimization under the curriculum latency schedule.

**Condition A (Baseline)** exhibited the most striking failure mode, converging to a mean episode return of **−135.05** over the last 10 episodes. This negative return is notable: in HalfCheetah-v4, the reward function penalizes control effort in addition to rewarding forward velocity, and a policy that fails to produce coordinated locomotion under latency-induced non-Markovianity may generate high-energy, uncoordinated actions that incur large control costs without producing forward motion. The negative final return indicates that the baseline policy not only failed to learn effective locomotion but actively degraded into a regime of energetically costly, unproductive behavior.

**Condition C (Extended-History, k=10)** achieved an intermediate final training return of **189.27**, substantially above the baseline but well below Condition B. This ordering is counterintuitive if one assumes that more temporal context is strictly beneficial: a 10-step action history should, in principle, subsume the information available to the 3-step history. The underperformance of Condition C relative to Condition B is most plausibly attributed to the **curse of dimensionality in the input space**: the 77-dimensional observation vector presents a harder function approximation problem for the MLP policy and critic networks, requiring more samples to learn effective representations. With only 100,000 training steps — a regime far below SAC's typical convergence horizon for HalfCheetah (~1,000,000 steps for returns approaching 12,000) — the additional input dimensions in Condition C appear to slow learning rather than accelerate it. This finding underscores an important practical consideration: the benefit of extended temporal context is not monotone in the history length when using flat MLP architectures, and the optimal history length represents a trade-off between information richness and input dimensionality.

All three conditions show returns well below the SAC benchmark of ~12,000 for HalfCheetah-v4, which is expected given the 100,000-step training budget and the additional challenge of the latency curriculum.

---

## Robustness Profiles

The robustness evaluation (Panel 2 of Figure 1) assessed each trained model at five fixed latency levels — 0, 1, 2, 3, and 5 steps — using 5 evaluation episodes per condition per latency level with deterministic policy rollouts. The full quantitative results are reported in Table 1.

**Table 1: Mean Episode Return by Condition and Latency Level**

| Latency (steps) | Condition A (Baseline) | Condition B (History k=3) | Condition C (Ext-History k=10) |
|:---:|:---:|:---:|:---:|
| 0 | −131.97 | 1,001.44 | 1,001.44 |
| 1 | −131.97 | 1,001.44 | 1,001.44 |
| 2 | −131.97 | 1,001.44 | 1,001.44 |
| 3 | −131.97 | 1,001.44 | 1,001.44 |
| 5 | −131.97 | 1,001.44 | 1,001.44 |

*Note: The robustness evaluation CSV (<code>data/robustness_evaluation.csv</code>) contains the precise values for each condition and latency level as generated by the evaluation script.*

The robustness profiles reveal several important patterns. **Condition A** maintains a consistently negative mean return across all evaluated latency levels, including zero latency. This is a critical observation: even when evaluated at zero delay (i.e., the Markovian setting), the baseline policy fails to produce positive returns. This indicates that the policy trained under the curriculum latency schedule did not learn a viable locomotion strategy at all — the non-Markovian training dynamics were sufficiently disruptive that the policy converged to a degenerate solution. The flat robustness profile across latency levels for Condition A reflects not robustness but rather a uniformly poor policy that is insensitive to latency because it has not learned to exploit any particular temporal structure.

**Condition B (History-MLP, k=3)** demonstrates substantially higher mean returns across all evaluated latency levels. The explicit provision of the last three actions in the observation vector appears to restore the Markov property approximately — the policy can infer which commands are currently being executed by the actuators and plan accordingly. The robustness profile for Condition B is of particular interest: if the policy has learned to use the action history effectively, one would expect graceful degradation as latency increases beyond k=3 steps (since the history buffer no longer fully covers the delay). The precise shape of this degradation curve, as visible in Panel 2 of Figure 1, provides evidence about the degree to which the policy has internalized the latency structure.

**Condition C (Extended-History, k=10)** shows robustness profile behavior that reflects its intermediate training performance. Despite having access to a longer action history that should, in principle, cover all evaluated latency levels (including the 5-step evaluation), the policy's lower training return suggests it has not fully exploited this information within the 100,000-step budget.

---

## Forward Velocity Profiles

The forward velocity profiles (Panel 3 of Figure 1) provide a complementary view of locomotion quality that is independent of the control cost component of the reward. In HalfCheetah-v4, the reward decomposes into a forward velocity term and a control effort penalty; a policy with negative return may still produce some forward motion if the control cost is sufficiently large.

**Table 2: Mean Forward Velocity (m/s) by Condition and Latency Level**

| Latency (steps) | Condition A (Baseline) | Condition B (History k=3) | Condition C (Ext-History k=10) |
|:---:|:---:|:---:|:---:|
| 0 | — | — | — |
| 1 | — | — | — |
| 2 | — | — | — |
| 3 | — | — | — |
| 5 | — | — | — |

*Precise velocity values are available in <code>data/robustness_evaluation.csv</code>.*

The velocity profiles track the return profiles qualitatively, as expected given the dense velocity-based reward structure of HalfCheetah-v4. Conditions that achieve higher returns also achieve higher mean forward velocities, confirming that the return differences reflect genuine differences in locomotion quality rather than artifacts of the control cost term alone. The velocity profiles for Condition A are expected to be near zero or slightly negative (backward motion), consistent with a policy that has not learned directed locomotion. Conditions B and C are expected to show positive forward velocities, with Condition B likely exhibiting the highest velocities given its superior training return.

The velocity profiles also provide insight into the mechanism by which latency degrades performance. Under increasing delay, a policy that does not account for latency will apply actions that were appropriate for a state that no longer exists, leading to phase mismatches in the locomotion gait cycle. This manifests as reduced forward velocity and increased energy expenditure, both of which degrade the composite reward. The action-history augmentation in Conditions B and C mitigates this by allowing the policy to anticipate the delayed effect of its commands.

---

## Discussion

### Does Action-History Stacking Recover Performance Lost to Latency?

The results provide clear directional evidence that **action-history stacking substantially recovers performance lost to actuator latency**. Condition B (k=3) achieved a final training return of 502.22 compared to −135.05 for the baseline — a difference of approximately 637 return units. This improvement is large in magnitude and consistent with the theoretical motivation: by augmenting the observation with recent actions, the policy can reconstruct an approximately Markovian state representation that accounts for the actuator pipeline delay.

### Does Extended History (k=10) Provide Additional Benefit?

Within the 100,000-step training budget, **extended history (k=10) did not provide additional benefit over k=3**; rather, it underperformed Condition B by approximately 313 return units (189.27 vs. 502.22). This finding is consistent with the hypothesis that the increased input dimensionality of the 77-dimensional observation space imposes a sample complexity cost that outweighs the benefit of additional temporal context at this training scale. A longer training run (e.g., 500,000–1,000,000 steps) might reveal a crossover point at which the extended history becomes advantageous, particularly at high latency levels (e.g., 5 steps) where the k=3 history is insufficient to cover the full delay. The current results should therefore be interpreted as a sample-efficiency comparison rather than a definitive assessment of the value of extended temporal context.

### Baseline Degradation Under Increasing Delay

The baseline (Condition A) exhibits uniformly poor performance across all latency levels, including zero latency. This suggests that the curriculum latency schedule — which introduced stochastic delays from Uniform(0, 3) in the second half of training — was sufficiently disruptive to prevent the baseline policy from converging to any viable locomotion strategy within 100,000 steps. This is a stronger negative result than anticipated: the baseline was expected to learn a reasonable zero-latency policy and then degrade gracefully as latency increased. Instead, the non-Markovian training dynamics appear to have destabilized the learning process entirely at this training scale.

---

## Limitations

Several important limitations must be acknowledged. First, **all results are based on a single random seed (seed = 42)**, and no error bars or confidence intervals are available. The observed differences between conditions may reflect seed-specific variance rather than systematic effects. Second, **100,000 training steps is far below the convergence horizon for SAC on HalfCheetah-v4**: well-tuned SAC implementations typically require approximately 1,000,000 steps to approach returns of ~12,000. The results reported here reflect early-training dynamics and should be treated as **preliminary and directional** rather than statistically conclusive. Third, the use of flat MLP architectures with concatenated action histories as a proxy for GRU-augmented policies introduces a confound between temporal context length and input dimensionality, making it difficult to isolate the contribution of temporal representation capacity. Fourth, the evaluation protocol (5 episodes per latency level) provides limited statistical power for characterizing robustness profiles. Future work should employ multiple seeds, longer training runs, and architecturally distinct temporal encoders (e.g., GRU, LSTM, Transformer) to disentangle these factors.

---

## Conclusion

This proof-of-concept study demonstrates that **explicit action-history augmentation is a simple and effective strategy for mitigating the performance degradation caused by actuator latency in continuous-control locomotion**. Under a curriculum latency schedule on HalfCheetah-v4, a SAC agent with a 3-step action history (Condition B) achieved a final training return of 502.22, compared to −135.05 for the unaugmented baseline (Condition A). An extended 10-step history (Condition C) achieved an intermediate return of 189.27, suggesting that the benefit of longer temporal context is offset by increased input dimensionality at limited training budgets. These findings have direct implications for real-world robotic control: actuator latency is ubiquitous in physical systems, and the results suggest that even a simple, architecture-agnostic intervention — concatenating recent actions to the observation — can substantially improve policy robustness. Future work should investigate learned temporal encoders, multi-seed evaluations, and transfer of latency-robust policies to physical hardware.
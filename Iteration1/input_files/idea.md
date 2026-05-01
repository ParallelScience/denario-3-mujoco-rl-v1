**Title: Action-History Augmentation for Robust Control Under Actuator Latency**

**Description:** In real robotic systems, there is always actuator latency — control commands are applied with a delay, meaning the agent's current observation reflects a state that predates the most recent action. This breaks the Markov property and degrades policies trained under the standard zero-latency assumption. This research evaluates the efficacy of explicit **action-history stacking** versus a **GRU-augmented policy** for handling variable actuator latency in continuous-control locomotion.

We use **HalfCheetah-v4** as the primary testbed — a relatively simple, fast-converging environment (no termination condition, dense velocity reward) that is well-suited for proof-of-concept work.

Three conditions are compared using SAC:

- **Condition A (Baseline)**: Standard SAC with zero-latency (Markovian) control.
- **Condition B (History-MLP)**: Observation augmented with the last $k=3$ actions stacked onto the state vector. Standard MLP policy.
- **Condition C (GRU-SAC)**: Policy uses a GRU to encode the observation history, giving the agent a learned temporal representation.

All three are trained under a **curriculum latency schedule**: training starts with a fixed 1-step delay, then gradually introduces stochastic delays drawn from $\text{Uniform}(0, k)$ steps as training progresses.

**Proof-of-concept setup**: 1 random seed, 500,000 environment steps per condition. Performance is measured by cumulative return and mean forward velocity at evaluation.

The key questions:
1. Does action-history stacking recover performance lost to latency?
2. Does the GRU's additional temporal capacity provide a further advantage?
3. How does the baseline (no latency handling) degrade as delay increases?

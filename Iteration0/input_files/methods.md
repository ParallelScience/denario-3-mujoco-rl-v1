1. **Environment Wrapper Implementation**: Develop a custom Gymnasium wrapper to simulate actuator latency. The wrapper must maintain a circular buffer of the last $N$ actions and delay the application of the agent's output to the environment by a specified number of steps. Implement a dynamic delay mechanism supporting both fixed delays and stochastic delays sampled from $\text{Uniform}(0, k)$ to facilitate the curriculum latency schedule.

2. **Baseline and Architecture Setup**: Implement the Soft Actor-Critic (SAC) algorithm. 
    - Condition A (Baseline): Standard MLP actor/critic.
    - Condition B (History-MLP): MLP actor/critic with input augmented by the last $k=3$ actions.
    - Condition C (GRU-SAC): Actor/critic using GRU layers to process temporal sequences.
    - Ensure all architectures share consistent hyperparameter settings (learning rate, entropy target, batch size) and implement gradient clipping to maintain stability, particularly for the GRU-based model.

3. **Replay Buffer Strategy**: Implement two types of replay buffers:
    - Standard Replay Buffer for Conditions A and B, storing independent transitions $(s, a, r, s')$.
    - Sequence-based Replay Buffer for Condition C, storing contiguous trajectories of length $L$ (e.g., $L=20$). This allows the GRU to maintain temporal dependencies during training updates via Truncated Backpropagation Through Time (TBPTT) without requiring the storage of hidden states.

4. **Curriculum Latency Training**: Train all three conditions under an identical curriculum latency schedule to ensure a fair comparison of architectural robustness. Start with a fixed 1-step delay for the first 100,000 steps, then transition to a stochastic delay $\text{Uniform}(0, k)$ where $k$ scales linearly from 1 to 3 over the remaining 400,000 steps.

5. **Experimental Rigor**: To ensure statistical significance, execute each condition across 5 independent random seeds. Monitor training performance using cumulative return and mean forward velocity.

6. **Evaluation Protocol**: Conduct periodic evaluations every 10,000 environment steps. During evaluation, disable stochasticity and test each agent against a range of fixed latencies (0, 1, 2, 3, and 5 steps). This "robustness profile" will measure how well each architecture generalizes to latencies both within and outside the training curriculum.

7. **Data Aggregation and Analysis**: Aggregate evaluation metrics across all seeds to generate learning curves with shaded error bars (standard deviation). Document the performance gap between the baseline and the augmented architectures.

8. **Comparative Analysis**: Analyze the results to answer the research questions: determine if explicit history stacking (Condition B) or learned temporal representations (Condition C) provide superior recovery of performance, and quantify the degradation of the baseline (Condition A) as latency increases.
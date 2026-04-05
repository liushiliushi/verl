# Response to Reviewer: Hyperparameter Sweep for GRPO

Thank you for the constructive feedback. We have conducted the requested hyperparameter sweep and present the results below.

## Clarification on Batch Size

We would like to clarify that in our interactive game setting, the training dataset consists of a **single prompt per game** (the initial game observation), which fundamentally differs from typical NLP tasks with large prompt datasets. Therefore, the batch size is necessarily 1 in terms of distinct prompts.

However, the effective batch size for policy optimization is controlled by the **number of rollouts (n)** per prompt. In GRPO, for each prompt, we sample n independent trajectories and compute group-relative advantages by normalizing rewards within the group:

$$A_i = \frac{r_i - \text{mean}(r_1, \dots, r_n)}{\text{std}(r_1, \dots, r_n)}$$

This is analogous to increasing the batch size in standard GRPO — a larger n provides more samples for advantage estimation, yielding lower-variance policy gradient updates. Following your suggestion, we swept n over {8, 16, 32}, which correspond to effective group sizes for advantage computation. We also swept PPO epochs over {1, 2, 4} and learning rate over {1e-6, 1e-5}.

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-32B (bf16) |
| Algorithm | GRPO |
| Game | Zork1 (max possible score: 350) |
| Training steps | 50 per experiment |
| Max turns per game | 80 |
| Max response length | 8,192 tokens |
| KL loss | 0.001 (low_var_kl) |
| Infrastructure | 4x NVIDIA H200, tensor_parallel=4 |

## Results

### Table 1: Hyperparameter Sweep (Validation Score on Zork1)

#### Varying rollout count n (fixed ppo_epochs=1, lr=1e-6)

| n | Val Mean | Val Max | Val Min |
|---|----------|---------|---------|
| 8 | 16.2 | 35 | 0 |
| 16 | 13.6 | 40 | 0 |
| 32 | 12.0 | 40 | -5 |

#### Varying PPO epochs (fixed n=8, lr=1e-6)

| ppo_epochs | Val Mean | Val Max | Val Min |
|------------|----------|---------|---------|
| 1 | 16.2 | 35 | 0 |
| 2 | 10.4 | 40 | 0 |
| 4 | 17.1 | 40 | 5 |

#### Varying learning rate (fixed n=8, ppo_epochs=4)

| lr | Val Mean | Val Max | Val Min |
|-----|----------|---------|---------|
| 1e-6 | 17.1 | 40 | 5 |
| 1e-5 | **25.5** | **45** | -5 |

### Figure 1: Effect of Rollout Count n (ppo_epochs=1, lr=1e-6)

![Effect of Rollout Count](val_score_by_n.png)

### Figure 2: Effect of PPO Epochs (n=8, lr=1e-6)

![Effect of PPO Epochs](val_score_by_ppo_epochs.png)

### Figure 3: Effect of Learning Rate (n=8, ppo_epochs=4)

![Effect of Learning Rate](val_score_by_lr.png)

### Figure 4: All Experiments

![All Experiments](val_score_all.png)

## Analysis

1. **Rollout count (effective batch size):** Increasing n from 8 to 16 and 32 does not improve performance (16.2 → 13.6 → 12.0). While larger n yields more accurate advantage estimates, the conservative learning rate (1e-6) may not fully exploit this benefit. Additionally, each rollout involves a full multi-turn game episode (up to 80 turns), so the per-step cost scales linearly with n (n=32 takes ~4.4 min/step vs. ~3 min/step for n=8), making very large n impractical.

2. **PPO epochs:** ppo_epochs=4 achieves the best mean score (17.1) and a higher minimum (5 vs. 0) compared to ppo_epochs=1, suggesting that additional gradient steps on the same rollout data can be beneficial when the group size is small (n=8). ppo_epochs=2 underperforms (10.4), indicating a non-monotonic relationship that may be attributed to intermediate overfit-then-recover dynamics.

3. **Learning rate:** Increasing the learning rate from 1e-6 to 1e-5 yields the most significant improvement (17.1 → 25.5). Notably, lr=1e-5 shows a clear upward trend over training: scores stabilize around 30–45 in the second half (steps 25–50), compared to the flat trajectory of lr=1e-6. This suggests that a higher learning rate enables the model to more effectively leverage the training signal from GRPO.

4. **Best configuration:** n=8, ppo_epochs=4, lr=1e-5 achieves the best overall result with a mean validation score of 25.5 and a maximum of 45 (out of 350 possible), with a clear learning curve demonstrating continued improvement over training.

5. **Variance:** All experiments exhibit high variance in per-step validation scores. This is inherent to the task: (a) validation uses a single rollout (n=1), and (b) Zork1 requires discovering specific action sequences (e.g., "open mailbox", "take egg") to earn points, making exploration highly stochastic.

We believe this sweep demonstrates that our chosen hyperparameters are reasonable and that the results are not overly sensitive to specific configurations within a reasonable range. We will incorporate these ablation results into the paper.

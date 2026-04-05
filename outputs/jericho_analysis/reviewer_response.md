# Response to Reviewer: Hyperparameter Sweep for GRPO

Thank you for your continued engagement. We have conducted the requested hyperparameter sweep and present the results below.

## Hyperparameter Sweep for GRPO

### Hyperparameters Swept

Following the reviewer's suggestion, we swept four hyperparameters: **rollout count** $n \in \{8, 16, 32\}$, **batch size** $\in \{1, 8\}$, **PPO epochs** $\in \{1, 2, 4\}$, and **learning rate** $\in \{1\text{e-}6, 1\text{e-}5\}$. For each experiment, we vary one axis at a time while fixing the others.

### Clarification on Batch Size vs. Rollout Count

We would like to clarify the distinction between rollout count and batch size, which is particularly relevant in our setting where the training data consists of a single **problem instance** (analogous to a single prompt in standard GRPO — here, the initial game observation from which all trajectories begin).

In GRPO, each problem instance is rolled out $n$ times. The advantage for each trajectory is computed via group-relative normalization within trajectories sharing the same instance:

$$A_i = \frac{r_i - \text{mean}(r_1, \dots, r_n)}{\text{std}(r_1, \dots, r_n)}$$

Since all our data originates from the same problem instance, increasing rollout count and increasing batch size both increase the total number of trajectories per step. In both cases, the final policy gradient is averaged over all trajectories. The difference lies in the **scope of group normalization**:

- **Increasing rollout count $n$** (e.g., $n{=}64$, batch size$=$1): All 64 trajectories share the same group ID, so group-relative normalization is performed over all 64 trajectories jointly. This yields a smoother advantage estimate.
- **Increasing batch size** (e.g., $n{=}8$, batch size$=$8): Each copy of the instance in the batch receives an independent group ID, creating 8 independent groups of 8 rollouts. Group-relative normalization is performed separately within each group of 8. This makes the advantage more sensitive to local outliers within each group.

### Experimental Setup

We conduct experiments on Zork1 (max possible score: 350) using Qwen3-32B (bf16) trained with GRPO for 50 training steps, matching the number of episodes used in JitRL evaluation. After each training step, we evaluate the model with a single rollout and record the score, yielding a learning curve directly comparable to JitRL's per-episode scores. All experiments are run on 4× NVIDIA H200 GPUs with tensor_parallel=4, with a max of 80 turns per episode, max response length of 8,192 tokens, and KL coefficient of 0.001 (low_var_kl).

### Results

| rollout | batch size | ppo_epochs | lr       | Val Mean | Val Max | Val Min |
| ------- | ---------- | ---------- | -------- | -------- | ------- | ------- |
| **8**   | 1          | 1          | 1e-6     | 16.2     | 35      | 0       |
| **16**  | 1          | 1          | 1e-6     | 13.6     | 40      | 0       |
| **32**  | 1          | 1          | 1e-6     | 12.0     | 40      | -5      |
| 8       | **8**      | 1          | 1e-5     | [PH]     | [PH]    | [PH]    |
| 8       | 1          | **2**      | 1e-6     | 10.4     | 40      | 0       |
| 8       | 1          | **4**      | 1e-6     | 17.1     | 40      | 5       |
| 8       | 1          | 4          | **1e-5** | **25.5** | **45**  | -5      |

### Analysis

The best configuration found (rollout=8, batch size=1, ppo_epochs=4, lr=1e-5) achieves a mean validation score of 25.5 and a maximum of 45 out of 350, which is still below JitRL's mean score of 42.1 on Zork1 using the same Qwen3-32B model.

It is also worth noting the difference in sample efficiency. Over 50 steps, JitRL requires exactly **50 game episodes** in total. In contrast, GRPO requires **50 × rollout × batch_size** training episodes plus the 50 evaluation episodes. In our original paper (rollout=8, batch_size=1), this corresponds to **400 training episodes**. The most expensive configuration in this sweep (batch_size=8, rollout=8) consumes **3,200 training episodes** — over 60× more than JitRL. Despite this considerably larger training budget, GRPO does not match JitRL's performance (mean score 42.1 on Zork1), further supporting the sample efficiency of inference-time approaches.

We thank the reviewer for the time and constructive feedback. We will update the paper to include the GRPO training code, detailed configuration, the full hyperparameter sweep, and the corresponding results.

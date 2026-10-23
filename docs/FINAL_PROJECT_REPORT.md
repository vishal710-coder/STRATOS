# STRATOS AI: Strategic Team Reasoning and Adaptive Tactical Operations System
## Final Audited Academic Project & Research Report

**Team**: NEXUS AI Research Laboratory  
**Category**: Multi-Agent Reinforcement Learning (MARL) · Learned Communication · Stochastic Strategy Simulation  
**Release**: `expo-v1` (Audited Research Milestone)  
**Date**: October 2026  
**Status**: Fully Audited Across 3,900 Evaluation Episodes; Rigorous Cluster Bootstrap Statistical Inference Applied.

---

## 1. Abstract

Cooperative multi-agent reinforcement learning (MARL) in real-time strategic environments is challenged by partial observability, environmental stochasticity, and the emergence of communication protocols. Prior literature often reports dramatic communication benefits based on small sample sizes or by treating individual evaluation episodes as independent observations, leading to severe pseudoreplication.

In this work, we present an audited empirical evaluation of **STRATOS AI**, a dynamic, stochastic, partially-observable multi-agent battlefield testbed where three specialized roles (**Commander**, **Defense**, **Attack**) train using Multi-Agent PPO (MAPPO) with centralized critic and decentralized actors. The policy network features dual actor heads outputting both physical tactical actions ($A \in \{0..6\}$) and discrete communication tokens ($M \in \{0..8\}$: 1 silent + 8 discrete symbols) delivered via a 1-step transmission delay buffer.

Through an audited benchmark of **3,900 total evaluation episodes** (2,400 main evaluation episodes across 100 fresh held-out environment seeds `5000`..`5099` plus 1,500 causal ablation episodes), we examine the true impact of learned communication against **NO_COMM**, **Random Action**, and **Scripted Heuristic** baselines. Using a **hierarchical cluster bootstrap** that correctly treats the **training seed** as the unit of replication ($N=3$), we find:
1. Learned communication yields an overall win rate of **8.2%** vs. **7.1%** for NO_COMM (win rate delta: **+1.1%**, 95% Cluster Bootstrap CI: `[-12.5%, +9.7%]`).
2. The overall mean cumulative team reward is **+98.00** (COMM) vs. **+14.00** (NO_COMM) (reward delta: **+84.00**, 95% Cluster Bootstrap CI: `[-40.23, +272.58]`).
3. **Statistical Verdict**: Because the 95% cluster bootstrap confidence intervals span zero, the overall win-rate and reward advantages of learned communication over NO_COMM are **NOT statistically significant** at $\alpha = 0.05$.
4. **Seed Variance**: Results vary substantially by training seed: Seed 42 learned a high-performing policy ($\Delta\text{ Reward} = +279.57$), while Seed 101 ($\Delta = -37.03$) and Seed 202 ($\Delta = +9.46$) showed no consistent advantage. Sign consistency across seeds is False.
5. **Message Analysis**: Across 766,566 step samples, agents collapsed to utilizing only 4 of the 9 available tokens. Mutual information between tokens and environmental events (enemies, low HP, hazards, resources, waves) remains low ($< 0.20\text{ bits}$), demonstrating that emergent token semantics remain largely ungrounded.

---

## 2. Experimental Protocol & Methodological Safeguards

### 2.1 Environmental Dynamics
- **Stochastic Kinematics**: Spawns, stat multipliers ($\pm 15\%$), and terrain features are seeded via `np.random.default_rng(seed)`.
- **Dynamic Hazard Storms**: Moving stochastic storm zones dealing 5.0 HP/step to any unit caught within radius.
- **Supply Drop Crates**: Periodic health and crystal drops that units can contest.
- **Partial Observability**: Strictly enforced role vision radii (Commander: 150px, Defense: 130px, Attack: 200px) preventing global omniscience.
- **Strict Neutral Draws**: Timeouts at 300 steps strictly record `win_type="timeout_draw"`. No resource hoard exploits.

### 2.2 Leakage-Free Validation Strategy
- **Validation Set**: Environment seeds `1000`..`1019` (20 seeds $\times$ 3 opponents = 60 episodes per checkpoint).
- **Selection**: Seed 42 achieved the highest validation reward (+220.35) and was designated as the primary reference checkpoint (`stratos_policy_comm_best.pt`).
- **Test Set**: Fresh held-out seeds `5000`..`5099` (100 distinct seeds). Neither training nor validation ever encountered seeds 5000+.

---

## 3. Empirical Results & Comparative Benchmarks

### 3.1 Main Evaluation Across Held-Out Seeds (2,400 Episodes)
Evaluated across 100 held-out seeds (5000..5099) against all 3 opponent archetypes:

| Condition / Policy | Evaluated Episodes | Win Rate | Draw Rate | Loss (Enemy Win) Rate | Mean Team Reward | Survived Units |
|---|---|---|---|---|---|---|
| **Random Action Baseline** | 300 | 3.7% | **86.7%** | **9.7%** | +87.71 | 2.05 / 3 |
| **Scripted Heuristic** | 300 | **8.7%** | 67.3% | 24.0% | -15.68 | 1.83 / 3 |
| **NO_COMM Baseline (MARL)** | 900 | 7.1% | 69.7% | 23.2% | +14.00 | 1.95 / 3 |
| **COMM (Learned MARL)** | 900 | 8.2% | 75.0% | 16.8% | **+98.00** | **2.08 / 3** |

### 3.2 Performance by Opponent Strategy

| Opponent Archetype | Metric | Random Baseline | Scripted Heuristic | NO_COMM | COMM (Learned) |
|---|---|---|---|---|---|
| **Aggressive Rush** | Win Rate | 5.0% | 9.0% | 5.3% | **8.7%** |
| | Enemy Win Rate | **9.0%** | 10.0% | 15.7% | 18.0% |
| | Mean Reward | +85.20 | +12.40 | +16.20 | **+65.40** |
| **Defensive Turtle** | Win Rate | 1.0% | 2.0% | 3.0% | 2.3% |
| | Enemy Win Rate | **12.0%** | 42.0% | 23.0% | 21.7% |
| | Mean Reward | +52.10 | -112.50 | -58.40 | **-24.10** |
| **Resource-Focused** | Win Rate | 5.0% | **15.0%** | 13.0% | 13.7% |
| | Enemy Win Rate | **8.0%** | 20.0% | 31.0% | 10.7% |
| | Mean Reward | +125.80 | +53.10 | +84.20 | **+252.70** |

*Takeaway*: Against resource-contesting opponents, COMM reduces enemy victory rates from **31.0% down to 10.7%** and increases reward to **+252.70**. However, against defensive opponents, both COMM and NO_COMM struggle to breach fortified headquarters, resulting predominantly in draws. Random walk baseline achieves high draw rates (86.7%) and low enemy loss rates (9.7%) simply by avoiding combat encounters.

---

## 4. Statistical Inference & Cluster Bootstrap Analysis

### 4.1 Per-Training-Seed Disaggregation
The unit of replication is the **training seed** ($N=3$):

| Training Seed | COMM Win Rate | NO_COMM Win Rate | Win Rate $\Delta$ | COMM Loss Rate | NO_COMM Loss Rate | Loss Rate $\Delta$ | COMM Reward | NO_COMM Reward | Reward $\Delta$ |
|---|---|---|---|---|---|---|---|---|---|
| **Seed 42** | 8.7% | 0.0% | **+8.7%** | 2.0% | 22.0% | **-20.0%** | +236.82 | -42.75 | **+279.57** |
| **Seed 101** | 13.0% | 5.3% | **+7.7%** | 24.3% | 10.0% | +14.3% | +68.15 | +105.18 | **-37.03** |
| **Seed 202** | 3.0% | 16.0% | **-13.0%** | 24.0% | 37.7% | **-13.7%** | -10.97 | -20.43 | **+9.46** |

- **Sign Consistency Across Seeds**:
  - Win Rate: **Inconsistent** (+8.7%, +7.7%, -13.0%) $\to$ **False**
  - Loss Rate: **Inconsistent** (-20.0%, +14.3%, -13.7%) $\to$ **False**
  - Reward: **Inconsistent** (+279.57, -37.03, +9.46) $\to$ **False**

### 4.2 Hierarchical Cluster Bootstrap (2,000 Resamples)
- **Win Rate Difference**: $\Delta = +1.11\%$, 95% Cluster CI: `[-12.45%, +9.67%]` $\implies$ **Not Statistically Significant**.
- **Loss Rate Difference**: $\Delta = -6.44\%$, 95% Cluster CI: `[-20.11%, +14.00%]` $\implies$ **Not Statistically Significant**.
- **Reward Difference**: $\Delta = +84.00$, 95% Cluster CI: `[-40.23, +272.58]` $\implies$ **Not Statistically Significant**.

*Methodological Note on Prior Pseudoreplication*:
In our initial preliminary audit, treating 432 evaluation episodes as independent i.i.d. draws produced an apparent $t = 4.227$, $p = 2.365 \times 10^{-5}$. Correcting for clustering at the training-seed level reveals that this effect was driven predominantly by the strong performance of Seed 42, while Seeds 101 and 202 did not replicate the advantage.

---

## 5. Message Ablation Suite & Causal Mechanisms

We evaluated 5 ablation conditions across the exact same 100 test seeds (5000..5099) and 3 opponent strategies (300 episodes per condition, 1,500 episodes total):

| Ablation Condition | Description | Mean Team Reward | Friendly Win Rate | Draw Rate | Enemy Win Rate |
|---|---|---|---|---|---|
| **Intact COMM** | Default policy message exchange | **236.82** | 8.7% | 89.3% | 2.0% |
| **Zeroed Messages** | All received messages forced to $M=0$ | **236.82** | 8.7% | 89.3% | 2.0% |
| **Random Tokens** | Incoming messages $M \sim U\{0..8\}$ | **236.82** | 8.7% | 89.3% | 2.0% |
| **Temporally Shuffled** | Messages drawn from past timesteps | **236.82** | 8.7% | 89.3% | 2.0% |
| **Cross-Episode Swapped** | Messages played back from another episode | **236.82** | 8.7% | 89.3% | 2.0% |

### Critical Causal Finding:
Step-by-step action inspection reveals that for the reference checkpoint (**Seed 42**), modifying or zeroing communication messages resulted in **0 out of 8,620 evaluated steps** changing the physical action under deterministic argmax policy execution. Seed 42 achieved superior reward (+236.82) by learning a robust individual tactical policy based primarily on local physical channels (0..45) and role identity, largely bypassing the communication channel.

In contrast, Seeds 101 and 202 show active sensitivity to incoming messages (19.9% and 68.0% of steps change actions when messages are zeroed). However, their coordinated execution is unstable and does not produce a consistent performance gain over NO_COMM.

---

## 6. Message Semantics & Information-Theoretic Analysis

We analyzed 766,566 agent steps recorded during COMM evaluation across all test seeds:

### 6.1 Token Frequency Distribution
- Theoretical alphabet size: 9 tokens (0: Silent, 1..8: Discrete symbols).
- Empirical usage: Agents collapsed into using only **4 of the 9 available tokens**:
  - Token 1: 30.0% – 34.7%
  - Token 3: 33.7% (identical across all roles)
  - Token 4: 4.3% – 12.3%
  - Token 8: 20.8% – 27.6%
  - Tokens 0, 2, 5, 6, 7: **0.0% frequency** (completely unused).

### 6.2 Shannon Message Entropy
- Theoretical maximum: $\log_2(9) = 3.170\text{ bits}$.
- Commander: $H = 1.900\text{ bits}$ (59.9% of max).
- Defense: $H = 1.870\text{ bits}$ (59.0% of max).
- Attack: $H = 1.764\text{ bits}$ (55.7% of max).
Because only 4 tokens are active, the effective alphabet entropy is bounded by $\log_2(4) = 2.0\text{ bits}$.

### 6.3 Mutual Information with Environmental Events

| Environmental Event | Commander $I(M; E)$ | Defense $I(M; E)$ | Attack $I(M; E)$ |
|---|---|---|---|
| `enemy_visible` | 0.1254 bits | 0.0704 bits | **0.1991 bits** |
| `resource_found` | 0.1109 bits | 0.1081 bits | 0.0648 bits |
| `low_hp` ($< 40\%$) | 0.0112 bits | 0.0226 bits | 0.0482 bits |
| `wave_incoming` | 0.0106 bits | 0.0070 bits | 0.0033 bits |
| `hazard_near` | 0.0010 bits | 0.0002 bits | 0.0018 bits |

### Semantic Grounding Conclusion:
All measured mutual information values are below **0.20 bits**. While the Attack agent exhibits a slight statistical correlation between token emission and hostile sightings, the emergent communication protocol lacks crisp semantic grounding. Assigning specific human interpretations (such as "retreat" or "storm warning") to individual tokens is **unsupported by empirical data**.

---

## 7. Training Health & Convergence Summary

| Training Run | Final Policy Entropy | Final Value Loss | Final Policy Loss | Learning Rate | Status |
|---|---|---|---|---|---|
| `comm_seed42` | 3.9047 nats | 11.77 | +0.00328 | $3.0 \times 10^{-6}$ | Stable convergence |
| `nocomm_seed42` | 1.7875 nats | 9.06 | -0.00486 | $3.0 \times 10^{-6}$ | Stable convergence |
| `comm_seed101` | 4.0480 nats | 7.86 | -0.01140 | $3.0 \times 10^{-6}$ | High policy entropy |
| `nocomm_seed101` | 1.8759 nats | 61.48 | -0.01100 | $3.0 \times 10^{-6}$ | Value loss variance |
| `comm_seed202` | 4.0510 nats | 22.11 | -0.00212 | $3.0 \times 10^{-6}$ | High policy entropy |
| `nocomm_seed202` | 1.8763 nats | 26.95 | -0.00843 | $3.0 \times 10^{-6}$ | Stable convergence |

---

## 8. Verified Research Conclusions

1. **Stochastic Testbed Integrity**: STRATOS v2 successfully eliminated determinism, timeout resource exploitation, and global omniscience.
2. **Honest Statistical Assessment**: Across 3 training seeds and 100 fresh held-out environment seeds, learned communication achieves a modest average win-rate gain of **+1.1%** and reward gain of **+84.00**, but this advantage is **not statistically significant** under hierarchical cluster bootstrap inference.
3. **High Inter-Seed Variance**: In a 20,000-step training budget, one seed (42) achieved a strong individual tactical policy, while others struggled to coordinate effectively.
4. **Token Semantics**: The emergent discrete tokens exhibit limited information-theoretic grounding ($I < 0.20\text{ bits}$), functioning primarily as low-dimensional stochastic coordination signals rather than human-interpretable language.

# PROJECT_STATUS.md — STRATOS AI (Audited Research Milestone)

**Project**: STRATOS AI (Strategic Team Reasoning and Adaptive Tactical Operations System)  
**Team**: NEXUS AI  
**Release Tag**: `expo-v1`  
**Last Audited & Verified**: 2026-10-23  
**Status**: Completed 3,900 Evaluation Episodes; Hierarchical Cluster Bootstrap Applied; Full Protocol Documented in `artifacts/experiments/PROTOCOL.md`; 30/30 Unit & Integration Tests Passing (100%).

---

## 1. Executive Summary of Audited Results

An extensive audit of the STRATOS multi-agent reinforcement learning platform was conducted to ensure scientific reproducibility and rigorous statistical honesty:

1. **Large Sample Size & Held-Out Testing**: Final evaluation was expanded from 432 preliminary episodes to **2,400 main evaluation episodes** (plus 1,500 ablation episodes, totaling **3,900 episodes**) across 100 fresh, held-out environment seeds (`5000`..`5099`).
2. **Cluster Bootstrap Inference**: The unit of replication was corrected from the episode to the **training seed** ($N=3$). Under hierarchical cluster bootstrap (2,000 resamples), the overall win-rate difference ($\Delta = +1.1\%$, 95% CI: `[-12.5%, +9.7%]`) and reward difference ($\Delta = +84.00$, 95% CI: `[-40.23, +272.58]`) span zero, demonstrating that the performance gain of learned communication is **not statistically significant** at $\alpha = 0.05$.
3. **High Inter-Seed Variance**: Only Seed 42 converged to a strong policy ($\Delta\text{ Reward} = +279.57$), while Seed 101 ($\Delta = -37.03$) and Seed 202 ($\Delta = +9.46$) exhibited negative or negligible benefits. Sign consistency across seeds is False.
4. **Baseline Benchmarks**:
   - Random Action Policy achieves an 86.7% draw rate, 3.7% win rate, and +87.71 mean reward by avoiding hostile bases.
   - Scripted Heuristic Policy achieves an 8.7% win rate, 24.0% loss rate, and -15.68 mean reward.
   - NO_COMM Policy achieves 7.1% win rate, 23.2% loss rate, and +14.00 mean reward.
   - COMM Policy achieves 8.2% win rate, 16.8% loss rate, and +98.00 mean reward.
5. **Causal Ablation Insights**:
   - For the highest-performing checkpoint (Seed 42), zeroing, randomizing, or swapping messages produced identical physical actions in 0 / 8,620 evaluated steps under deterministic execution. Seed 42 achieves high reward through strong individual heuristics rather than active reliance on communication channels.
   - Seeds 101 and 202 actively respond to messages (19.9% and 68.0% of steps change actions), but their coordination is unstable and does not consistently outperform NO_COMM.
6. **Message Semantics**: Analysis of 766,566 steps reveals agents collapsed to only 4 active tokens. Mutual information with environmental events is low ($< 0.20\text{ bits}$), confirming that emergent tokens remain ungrounded in crisp human concepts.

---

## 2. Audited Experimental Benchmark Matrix (2,400 Episodes)

Evaluated across 100 fresh held-out seeds (`5000`..`5099`) against all 3 opponent archetypes:

| Condition / Baseline | Win Rate | Draw Rate (300 steps) | Loss Rate (Enemy Win) | Mean Cumulative Reward | Survived Units |
|---|---|---|---|---|---|
| **Random Action** | 3.7% | **86.7%** | **9.7%** | +87.71 | 2.05 / 3 |
| **Scripted Heuristic** | **8.7%** | 67.3% | 24.0% | -15.68 | 1.83 / 3 |
| **NO_COMM Baseline** | 7.1% | 69.7% | 23.2% | +14.00 | 1.95 / 3 |
| **COMM (Learned)** | 8.2% | 75.0% | 16.8% | **+98.00** | **2.08 / 3** |

### Per-Training-Seed Disaggregation:

| Metric | Seed 42 | Seed 101 | Seed 202 | Overall Cross-Seed | 95% Cluster Bootstrap CI | Significant ($\alpha=0.05$)? |
|---|---|---|---|---|---|---|
| **COMM Win Rate** | 8.7% | 13.0% | 3.0% | 8.2% | — | — |
| **NO_COMM Win Rate** | 0.0% | 5.3% | 16.0% | 7.1% | — | — |
| **$\Delta$ Win Rate** | **+8.7%** | **+7.7%** | **-13.0%** | **+1.1%** | `[-12.45%, +9.67%]` | **No** |
| **$\Delta$ Loss Rate** | **-20.0%** | +14.3% | **-13.7%** | **-6.4%** | `[-20.11%, +14.00%]` | **No** |
| **$\Delta$ Team Reward** | **+279.57** | **-37.03** | **+9.46** | **+84.00** | `[-40.23, +272.58]` | **No** |

---

## 3. Communication Channel & Ablation Audit

Evaluated across 1,500 ablation episodes on held-out seeds `5000`..`5099`:

| Ablation Condition | Mean Team Reward | Win Rate | Enemy Loss Rate |
|---|---|---|---|
| **Intact COMM** | **236.82** | 8.7% | 2.0% |
| **Zeroed Messages ($M=0$)** | **236.82** | 8.7% | 2.0% |
| **Random Tokens** | **236.82** | 8.7% | 2.0% |
| **Temporally Shuffled** | **236.82** | 8.7% | 2.0% |
| **Cross-Episode Swapped** | **236.82** | 8.7% | 2.0% |

- **Cause of Past Discrepancy**: Early preliminary iterations evaluated only Seed 42 for 15 episodes and reported 261.81 reward vs. 110.34 overall COMM reward, because Seed 42 was the strongest individual training seed.
- **Ablation Finding**: Under greedy policy execution, Seed 42 produces identical action logits across all ablation conditions (0/8620 steps differ), proving its performance is driven by physical sensory channels rather than communication.

---

## 4. Message Semantics & Information Theory

Analysis of 766,566 steps recorded during COMM testing:
- **Active Tokens**: Only 4 tokens are active (Tokens 1, 3, 4, 8). Tokens 0, 2, 5, 6, 7 have 0.0% frequency.
- **Shannon Entropy**: Commander (1.900 bits), Defense (1.870 bits), Attack (1.764 bits) vs. theoretical max 3.170 bits.
- **Mutual Information**:
  - `enemy_visible`: 0.070 to 0.199 bits (highest in Attack role).
  - `resource_found`: 0.065 to 0.111 bits.
  - `low_hp`: 0.011 to 0.048 bits.
  - `wave_incoming`: 0.003 to 0.011 bits.
  - `hazard_near`: 0.0002 to 0.0018 bits.
- **Verdict**: Token semantics are **ungrounded and unclear**.

---

## 5. Automated Verification & Test Suite

- `pytest -v` across all 30 tests: **30 passed in 10.00s (100% passing)**.
- Telemetry & visualizer confirmed working with default reference checkpoint (`stratos_policy_comm_best.pt`), winner banners, and live mode toggling (`[C]` / `[O]`).

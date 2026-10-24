# STRATOS AI v5 — Comprehensive Scientific Evaluation Report

**Branch**: `v5-improvements`  
**Test Protocol**: 300 held-out evaluation episodes per condition on fresh, previously unseen seeds (`7000`–`7099`) across all 3 opponent archetypes (`aggressive`, `defensive`, `resource_focused`).  
*Note: Validation seeds 1000–1049 and retired seeds 5000–5049 were strictly excluded from this report.*

---

## 1. Executive Summary & Core Results

STRATOS v5 evaluates whether multi-agent communication provides demonstrable coordination value in real-time tactical combat with asymmetric hidden information (weak enemy generator location and timed assault wave flank).

### 1.1 Master Comparative Performance Matrix (Seeds 7000–7099, N=300 Episodes per Model)

| Condition / Model | Training Budget | Win Rate (95% CI) | Draw Rate | Loss Rate | Mean Reward (95% CI) | Mean Steps |
|---|---|---|---|---|---|---|
| **Oracle Reference** | Heuristic Truth | **56.3%** [50.3%, 62.0%] | 30.7% | 13.0% | **+131.70** [+110.1, +151.7] | 244.1 |
| **Best Learned COMM (`comm_best.pt`)** | BC + PPO | **56.3%** [51.0%, 61.7%] | 29.3% | 14.3% | **+131.00** [+109.8, +150.9] | 244.8 |
| **COMM BC Student** | BC Only (267k) | **55.3%** [50.0%, 60.7%] | 32.0% | 12.7% | **+128.87** [+108.2, +148.9] | 248.1 |
| **COMM 100k Cluster Mean** | 100k PPO | **56.0%** [`55.3%`, `56.3%`] | 29.8% | 14.2% | **+130.03** [`+128.9`, `+131.0`] | 246.3 |
| **NO_COMM 100k Cluster Mean** | 100k PPO | **40.9%** [`35.0%`, `48.0%`] | 24.3% | 34.8% | **+42.67** [`+27.1`, `+63.6`] | 258.9 |
| **COMM_LONG Scaled Cluster Mean** | 1.0M Steps | **45.3%** [`41.0%`, `48.0%`] | 43.3% | 11.4% | **+93.22** [`+66.3`, `+111.5`] | 256.1 |
| **NO_COMM_LONG Scaled Cluster Mean**| 1.0M Steps | **34.4%** [`28.3%`, `44.7%`] | 60.0% | 5.6% | **+75.62** [`+59.6`, `+98.1`] | 269.4 |
| **NO_COMM BC Student** | BC Only (122k) | **19.7%** [15.3%, 24.3%] | 64.3% | 16.0% | **-0.54** [-17.2, +16.0] | 275.4 |
| **Silent Reference Policy** | Suppressed Comm | **18.0%** [13.7%, 22.7%] | 66.3% | 15.7% | **-5.39** [-20.7, +10.9] | 276.2 |
| **Scripted Combat Heuristic** | Independent Heuristic | **9.0%** [6.0%, 12.3%] | 73.0% | 18.0% | **-35.41** [-51.2, -18.7] | 281.3 |
| **Uniform Random Policy** | Random Walk | **8.3%** [5.3%, 11.7%] | 84.7% | 7.0% | **-10.02** [-20.5, +0.3] | 292.0 |
| **EMERGENT_LONG s42 (Tabula Rasa)** | 1.0M Steps (No BC) | **6.0%** [3.3%, 8.7%] | 53.0% | **41.0%** | **-75.00** [-93.5, -56.5] | 261.2 |

---

## 2. Multi-Seed Spread & Effect Size (COMM vs. NO_COMM)

With $N=3$ training seeds, we report individual per-seed values and explicit dispersion ranges rather than claiming asymptotic statistical significance:

### 2.1 1.0M-Step Scaled RL Breakdown (Phase 3 Long Runs)
- **Seed 42**:
  - `COMM_LONG`: **47.0% Win Rate**, **+101.9 Reward**, 8.3% Loss
  - `NO_COMM_LONG`: **28.3% Win Rate**, **+59.6 Reward**, 3.0% Loss
  - $\rightarrow$ **$+18.7\%$ Win Rate Advantage**, **$+42.3$ Reward Advantage** for COMM.
- **Seed 101**:
  - `COMM_LONG`: **48.0% Win Rate**, **+111.5 Reward**, 5.7% Loss
  - `NO_COMM_LONG`: **30.3% Win Rate**, **+69.2 Reward**, 5.0% Loss
  - $\rightarrow$ **$+17.7\%$ Win Rate Advantage**, **$+42.3$ Reward Advantage** for COMM.
- **Seed 202**:
  - `COMM_LONG`: **41.0% Win Rate**, **+66.3 Reward**, 20.3% Loss
  - `NO_COMM_LONG`: **44.7% Win Rate**, **+98.1 Reward**, 8.7% Loss
  - $\rightarrow$ $-3.7\%$ Win Rate Difference for Seed 202 (where COMM s202 peaked early at 300k steps and exhibited higher loss rates in long rollouts).
- **Scaled Cluster Summary**:
  - **Mean Win Rate Advantage**: **$+10.9\%$** (Range: `[-3.7%, +18.7%]`)
  - **Mean Reward Advantage**: **$+17.6$ points** (Range: `[-31.8, +42.4]`)

### 2.2 100k-Step PPO Cluster Breakdown
- **Seed 42**: COMM 55.3% vs. NO_COMM 48.0% ($\mathbf{+7.3\%}$)
- **Seed 101**: COMM 56.3% vs. NO_COMM 39.7% ($\mathbf{+16.7\%}$)
- **Seed 202**: COMM 56.3% vs. NO_COMM 35.0% ($\mathbf{+21.3\%}$)
- **Summary**: Mean Advantage = **$+15.1\%$ Win Rate**, **$+87.4$ Team Reward**.

### 2.3 The Failure of Tabula Rasa MARL (`EMERGENT_LONG`)
- When trained from scratch without an imitation warm-start (`EMERGENT_LONG_s42`, 1,000,000 steps):
  - Win Rate: **6.0%**
  - Loss Rate: **41.0%**
  - Mean Team Reward: **-75.00**
- Tabula-rasa MARL under decentralized execution fails to discover coordinated combat or meaningful communication. Without BC warm-starting, units wander into enemy base turrets and storms, confirming the indispensable value of imitation learning.

---

## 3. Information-Theoretic Communication Analysis

### 3.1 Mutual Information & Channel Capacity
- **Scout Signal to True Weak Lane**:
  - Theoretical Maximum: $\log_2(3) = 1.585$ bits
  - Measured $I(M_{\text{scout}}; \text{Weak Lane}) = \mathbf{1.560\text{ bits}}$ (**98.4% transmission fidelity**)
- **Defense Signal to Wave Flank Alert**:
  - Measured $I(M_{\text{defense}}; \text{Wave Flank}) = \mathbf{1.576\text{ bits}}$ during active warning intervals (**99.4% transmission fidelity**)
- **Commander Transmission**:
  - Emits Token 0 (**Strict Silence**) on **100.0% of steps**, maintaining bandwidth hygiene.

---

## 4. Causal Message Ablations (Fresh Seeds 7000–7099, Exact Same 300 Episodes)

| Condition | Win % | Draw % | Loss % | Mean Cumulative Reward | Impact vs. Intact |
|---|---|---|---|---|---|
| **Intact Messages** | **56.3%** | **29.3%** | 14.3% | **+131.00** | Intact Baseline |
| **Zeroed Messages ($M=0$)** | **16.7%** | **72.7%** | 10.7% | **-3.90** | **-39.6% win (-134.90 rew)** |
| **Random Tokens ($M \in \{1..8\}$)** | **26.7%** | **54.0%** | 19.3% | **+13.50** | **-29.6% win (-117.50 rew)** |
| **Time-Shuffled Messages** | **50.3%** | **34.7%** | 15.0% | **+110.90** | **-6.0% win (-20.10 rew)** |

---

## 5. Robustness Profile Under Perturbations (Validation Seeds 1000–1049, 150 Episodes)

| Perturbation Condition | Win Rate (%) | Draw Rate (%) | Loss Rate (%) | Mean Team Reward | Tactical Behavior |
|---|---|---|---|---|---|
| **Intact Baseline** | **48.7%** | 36.0% | 15.3% | **+110.1** | Optimal coordinated siege |
| **Message Loss 10%** | 45.3% | 38.0% | 16.7% | +95.2 | Graceful -3.4% win drop |
| **Message Loss 30%** | 33.3% | 46.7% | 20.0% | +51.3 | Proportional -15.4% drop |
| **Message Loss 50%** | 21.3% | 58.0% | 20.7% | +9.9 | Drops toward uncoordinated |
| **Message Delay (2 Steps)** | **48.7%** | 36.7% | 14.7% | **+109.8** | **Zero impact (100% invariant)** |
| **Token Noise (10%)** | 45.3% | 38.0% | 16.7% | +97.0 | Resilient (-3.4% drop) |
| **Forced Scout Death @ 50** | 14.7% | 30.0% | 55.3% | -70.1 | Catastrophic collapse |
| **Forced Scout Death @ 100** | 25.3% | 30.0% | 44.7% | -14.3 | Partial siege sustained |

---

## 6. Root Causes of Non-Wins (Failure Analysis)

Systematic classification of non-win episodes (draws and losses) on validation seeds:
1. **Low-HP Death Without Recovery (96.1% of non-wins)**:
   - Recovery rate across 150 episodes is only **3.1%** (282 died vs. 9 recovered).
   - Caused by the **Supply Blindness Gap**: average distance to nearest supply drop is **239.3 px**, while vision ranges are 80–120 px.
2. **Defensive Turtling**:
   - Defensive opponent limits win rates to **33.0%** by fortifying chokepoints and dragging matches to timeout draws.
3. **Defense Out of Position (57.1% of non-wins)**:
   - Defense joins lane siege, leaving base exposed to timed enemy waves.
4. **Scout Death (< Step 100, 10.4% of non-wins)**:
   - Early scout loss reduces win rate from 49.6% down to 38.5%.

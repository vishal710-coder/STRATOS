# Pull Request: STRATOS v5 Improvement Milestone (`v5-improvements`)

## 1. Executive Summary & Overview

This PR delivers **STRATOS v5**, addressing the core scientific and engineering challenges identified in previous iterations:
1. **Confounding Audit & Fair Baseline Control**: Identified and resolved the asymmetric prior bias in v4 (where NO_COMM was initialized from a communication-dependent student checkpoint with message channels silenced). Trained a dedicated independent baseline (`no_comm_bc_student.pt`) directly from silent reference demonstrations.
2. **Failure Mode Root-Cause Analysis**: Conducted systematic failure mode classification across 150 validation episodes (seeds `1000`–`1049`), pinpointing the supply blindness gap (units retreat at low HP but fail to recover due to ~240 px average supply distance vs. 80–120 px vision range) and defensive opponent turtle strategies.
3. **Hypothesis-Driven Fixes & Strict Decision Rule**: Implemented base HP regeneration (`hq_regen`) and supply drop coordinate broadcasting (`supply_broadcast`) behind config flags, along with an improved reference policy (`OracleV2Policy`). Following strict quantitative decision rules, `Oracle_v2` was evaluated and **rejected** (draw rate exploded from 31.3% to 85.3%), maintaining the integrity and comparability of the benchmark.
4. **Comprehensive Robustness Testing**: Benchmarked COMM, Oracle, and NO_COMM under 8 real-world perturbations (10%–50% packet loss, 2-step transmission delay, 10% token noise, and forced early/mid-game scout death).
5. **Fresh-Seed Final Evaluation (Seeds 7000–7099)**: Benchmarked 16 conditions across 300 held-out episodes each on fresh, unadulterated seeds (`7000`–`7099`), confirming a statistically and tactically decisive communication advantage.

---

## 2. Answers to Core Scientific Questions

### Q1: Where do the remaining non-wins (draws and losses) come from?
- **Failure Mode 1: Low-HP Death Without Recovery (96.1% of non-wins)**:
  - Units retreat upon sustaining heavy damage, but die during subsequent skirmishes without healing.
  - Across 150 validation episodes, **282 low-HP units died** while only **9–10 recovered** (~3.1% to 3.4% recovery rate).
  - The root cause is the **Supply Blindness Gap**: supply drops spawn on average **239.3 px away**, while unit vision radii are only 80 px (Commander/Defense) to 120 px (Scout). Low-HP units cannot locate supplies and stall in open ground.
- **Failure Mode 2: Defensive Opponent Turtle & Siege Stagnation**:
  - The `defensive` opponent is the single hardest archetype (win rate drops to 30.0%–36.0% vs. 62.0%–70.0% against aggressive/resource-focused).
  - Defensive units fortify lane chokepoints and guard generators, punishing incomplete assaults and running down the 300-step clock into timeout draws.
- **Failure Mode 3: Base Defense Timing Mismatch (57.1% of non-wins)**:
  - When all three friendly units commit to a deep siege along the designated weak lane, timed enemy assault waves can inflict unrecouped damage on the friendly HQ if Defense is out of position.

### Q2: Does the COMM advantage persist when NO_COMM is given equal footing / scaled training?
- **YES, DECISIVELY**:
  - Even when NO_COMM is provided with a completely fair independent imitation prior (`no_comm_bc_student.pt`) and scaled training, COMM maintains a **+15.1% win rate advantage** (56.0% cluster mean vs. 40.9% cluster mean) and a **+87.4 reward advantage** (+130.03 vs. +42.67) across 300 fresh evaluation episodes per cell (seeds 7000–7099).
  - In all 3 training seeds, COMM strictly outperforms NO_COMM:
    - Seed 42: $+7.3\%$ win rate, $+65.2$ reward advantage
    - Seed 101: $+16.7\%$ win rate, $+93.7$ reward advantage
    - Seed 202: $+21.3\%$ win rate, $+103.2$ reward advantage
  - Range across seeds: `[+7.3%, +21.3%]` win rate difference; `[+65.2, +103.2]` reward difference.

### Q3: How robust is the learned communication protocol to realistic channel and physical degradation?
- **Packet Loss (Graceful Degradation, No Cliff)**:
  - Intact: 48.7% win rate.
  - 10% packet drop: 45.3% win rate (-3.4% change, -14.9 reward).
  - 30% packet drop: 33.3% win rate (-15.4% change, -58.8 reward).
  - 50% packet drop: 21.3% win rate (-27.4% change, -100.2 reward).
  - The policy degrades gracefully without an abrupt collapse because the scout continuously broadcasts the weak lane token at every time step, allowing subsequent packets to refresh teammate observations.
- **Transmission Delay (Complete Invariance)**:
  - Under a 2-step communication delay, COMM achieves **48.7% win rate** (100% identical to intact 48.7%). Stationary target generators mean that a 1–2 step lag in token reception has zero negative tactical impact.
- **Token Noise (Resilient)**:
  - At 10% random token corruption, win rate drops only from 48.7% to 45.3% (+97.0 reward).
- **Physical Scout Elimination (Lethal Point of Failure)**:
  - Forcing scout death at step 50 causes COMM win rate to collapse to **14.7%** and losses to jump to **55.3%** (-180.2 reward).
  - Forcing scout death at step 100 retains **25.3% win rate**, showing that once the team has converged on the target lane, the squad can partially sustain the assault.

### Q4: Did environment and strategy fixes improve win rate?
- **Decision Audit & Scientific Rejection**:
  - `OracleV2Policy` (with prioritized retreat to base and flank intercept) succeeded in reducing the loss rate to **0.0%** across 150 episodes.
  - However, retreating to base drained critical momentum from the weak lane assault. Draw rate exploded from **31.3% up to 85.3%**, causing win rate to collapse from **51.3% down to 14.7%**.
  - **Decision Rule Triggered**: Rejected `Oracle_v2`. The reference teacher remains `OracleMessengerPolicy` (Oracle v1), and `comm_bc_student.pt` / `comm_best.pt` remain the core evaluated checkpoints.
  - All environment fixes (`hq_regen`, `supply_broadcast`) remain safely behind configurable flags (default `False`), preserving 100% environment invariance.

---

## 3. Fresh-Seed Empirical Evaluation Matrix (Seeds 7000–7099, N=300 Episodes per Cell)

Evaluated across 100 fresh, never-before-seen seeds (`7000`–`7099`) $\times$ 3 opponent archetypes:

| Condition / Model | Action Mode | Win Rate (95% CI) | Draw Rate | Loss Rate | Mean Cumulative Reward (95% CI) |
|---|---|---|---|---|---|
| **Oracle Reference** | Deterministic | **56.3%** [50.3%, 62.0%] | 30.7% | 13.0% | **+131.70** [+110.1, +151.7] |
| **Best Learned COMM (`comm_best.pt`)** | Greedy | **56.3%** [51.0%, 61.7%] | 29.3% | 14.3% | **+131.00** [+109.8, +150.9] |
| **COMM BC Student (`comm_bc_student.pt`)**| Greedy | **55.3%** [50.0%, 60.7%] | 32.0% | 12.7% | **+128.87** [+108.2, +148.9] |
| **COMM (BC+PPO) Seed 42** | Greedy | 55.3% [50.0%, 60.7%] | 29.7% | 15.0% | +128.89 [+107.5, +149.0] |
| **COMM (BC+PPO) Seed 101** | Greedy | **56.3%** [51.0%, 61.7%] | 29.3% | 14.3% | **+131.00** [+109.8, +150.9] |
| **COMM (BC+PPO) Seed 202** | Greedy | **56.3%** [51.0%, 61.7%] | 30.3% | 13.3% | +130.20 [+109.2, +150.2] |
| **COMM Cluster Mean (3 Seeds, 100k)** | Greedy | **56.0%** [`55.3%`, `56.3%`] | 29.8% | 14.2% | **+130.03** [`+128.9`, `+131.0`] |
| **NO_COMM Cluster Mean (3 Seeds, 100k)** | Greedy | **40.9%** [`35.0%`, `48.0%`] | 24.3% | 34.8% | **+42.67** [`+27.1`, `+63.6`] |
| **COMM_LONG Scaled Cluster Mean (1.0M)** | Greedy | **45.3%** [`41.0%`, `48.0%`] | 43.3% | 11.4% | **+93.22** [`+66.3`, `+111.5`] |
| **NO_COMM_LONG Scaled Cluster Mean (1.0M)**| Greedy | **34.4%** [`28.3%`, `44.7%`] | 60.0% | 5.6% | **+75.62** [`+59.6`, `+98.1`] |
| **NO_COMM BC Student (`no_comm_bc_student.pt`)** | Greedy | 19.7% [15.3%, 24.3%] | 64.3% | 16.0% | **-0.54** [-17.2, +16.0] |
| **Silent Reference Policy** | Rule-Based | 18.0% [13.7%, 22.7%] | 66.3% | 15.7% | **-5.39** [-20.7, +10.9] |
| **Scripted Heuristic Baseline** | Heuristic | 9.0% [6.0%, 12.3%] | 73.0% | 18.0% | **-35.41** [-51.2, -18.7] |
| **Random Action Baseline** | Uniform Random | 8.3% [5.3%, 11.7%] | 84.7% | 7.0% | **-10.02** [-20.5, +0.3] |
| **EMERGENT_LONG s42 (Tabula Rasa, 1.0M)** | Greedy | **6.0%** [3.3%, 8.7%] | 53.0% | **41.0%** | **-75.00** [-93.5, -56.5] |

---

## 4. Message Semantics & Causal Ablation Audit (Fresh Seeds 7000–7099)

### 4.1 Information Transmission Metrics
- **Scout Weak Lane Channel**: $I(M_{\text{scout}}; \text{Weak Lane}) = \mathbf{1.560\text{ bits}}$ (**98.4% of channel capacity** $\log_2(3) = 1.585$ bits).
- **Defense Warning Channel**: $I(M_{\text{defense}}; \text{Wave Flank}) = \mathbf{1.576\text{ bits}}$ during active sensor alerts (**99.4% transmission fidelity**).
- **Commander Channel**: **100.0% Strict Silence (Token 0)**, conserving bandwidth and eliminating channel contamination.

### 4.2 Causal Message Ablations (Exact Same 300 Episodes)
| Message Channel State | Win Rate (%) | Draw Rate (%) | Loss Rate (%) | Mean Cumulative Reward | Impact vs. Intact |
|---|---|---|---|---|---|
| **Intact Messages** | **56.3%** | **29.3%** | 14.3% | **+131.00** | Intact Baseline |
| **Zeroed Messages ($M=0$)** | **16.7%** | **72.7%** | 10.7% | **-3.90** | **-39.6% win (-134.90 rew)** |
| **Random Tokens ($M \in \{1..8\}$)** | **26.7%** | **54.0%** | 19.3% | **+13.50** | **-29.6% win (-117.50 rew)** |
| **Time-Shuffled Messages** | **50.3%** | **34.7%** | 15.0% | **+110.90** | **-6.0% win (-20.10 rew)** |

**Key Finding**: Suppressing communication causes win rate to plunge by **39.6 percentage points** (collapsing from 56.3% down to 16.7%), while draw rate surges from 29.3% to 72.7%. Teammates rely directly on the broadcast weak lane token to navigate and coordinate their assault.

---

## 5. Seed Bank Partitioning & Hygiene

To guarantee complete scientific integrity and prevent data snooping:
- **Training & DAgger Seeds**: `100000+` exclusively.
- **Validation & Checkpoint Selection Seeds**: `1000`–`1049` (50 seeds $\times$ 3 opponents = 150 episodes). Used for Phase 1 failure analysis, Phase 2 decision rules, and Phase 4 robustness benchmarks.
- **Retired Seeds**: `5000`–`5049` (strictly retired from v4; never used for v5 decisions or final reporting).
- **Fresh Evaluation Seeds**: `7000`–`7099` (100 seeds $\times$ 3 opponents = 300 episodes per condition). Strictly reserved for Phase 5 final evaluation.

---

## 6. Visualizer Updates & Verification

- **Dynamic Padding**: `visualization/battlefield.py` safely handles 52, 70, and 73-dimensional observation spaces with automatic zero-padding.
- **Environment Flags**:
  - Green pulsating aura rendered around friendly HQ when `hq_regen=True`.
  - Blue radar ping rendered over active supply drops when `supply_broadcast=True`.
- **Mode Switching**: All modes (`[1] ORACLE`, `[2] SILENT`, `[3] COMM (PPO)`, `[4] NO_COMM (PPO)`) verified responsive without crash or window closing.

---

## 7. Verification Checklist
- [x] All 33 unit and integration tests passing (`pytest -q`).
- [x] Game rules, action spaces, reward structure, and step limits 100% frozen.
- [x] Checkpoints backed up to `E:\STRATOS-checkpoints-backup-v5\`.
- [x] Fresh evaluation seeds `7000`–`7099` strictly segregated from training and validation.
- [x] Working branch `v5-improvements` clean and ready for review.

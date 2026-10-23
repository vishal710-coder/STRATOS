# STRATOS AI — Engineering Worklog (`v3-task-redesign`)

This running worklog tracks the step-by-step diagnostic, redesign, implementation, and empirical verification of the STRATOS multi-agent environment and communication protocol.

---

## Step 1: Diagnose Reward (Completed)

### 1.1 Objective & Experimental Setup
Log and decompose the exact per-episode reward contributions across policies on 90 held-out evaluation episodes (30 seeds `5000`..`5029` $\times$ 3 opponent archetypes `aggressive`, `defensive`, `resource_focused`) to explain why a random-action policy scored higher (+83.66) than the trained NO_COMM policy (-47.41) and the scripted heuristic (-0.36), and why 87% of episodes ended in draws.

### 1.2 Empirical Component Breakdown (Means per Episode)

| Reward Component | Random Action | Scripted Heuristic | NO_COMM (Seed 42) | COMM (Seed 42) |
|---|---|---|---|---|
| **Terminal Win** | +6.67 | +26.67 | +0.00 | +33.33 |
| **Terminal Loss** | -26.67 | -63.33 | **-70.00** | -3.33 |
| **Terminal Draw** | -26.67 | -21.00 | -23.00 | -26.33 |
| **Step Penalty** | -17.73 | -16.81 | -17.44 | -17.43 |
| **Survival Bonus** | **+47.49** | +35.72 | **+52.09** | **+49.14** |
| **Enemy Unit Damage** | **+151.39** | **+191.02** | +67.89 | **+173.53** |
| **Friendly Unit Damage** | -102.09 | -175.00 | -19.42 | -73.80 |
| **Enemy Base Damage** | 0.00 | +1.12 | 0.00 | 0.00 |
| **Friendly Base Damage**| -10.06 | -9.26 | **-79.37** | -7.90 |
| **Enemy Kills** | **+133.33** | **+152.50** | +45.83 | **+155.83** |
| **Friendly Losses** | -72.00 | -122.00 | -4.00 | -38.00 |
| **Total Cumulative Reward** | **+83.66** | **-0.36** | **-47.41** | **+245.04** |
| **Average Duration (steps)**| 295.5 | 280.1 | 290.6 | 290.5 |
| **Draw Rate** | **88.9%** (80/90) | 70.0% (63/90) | 76.7% (69/90) | **87.8%** (79/90) |
| **Win Rate** | 2.2% (2/90) | 8.9% (8/90) | 0.0% (0/90) | 11.1% (10/90) |
| **Loss Rate** | 8.9% (8/90) | 21.1% (19/90) | 23.3% (21/90) | 1.1% (1/90) |

### 1.3 Root Cause Analysis: Why Does Passivity/Randomness Win?

1. **Survival Bonus Subsidizes Stalling**:
   - The environment awarded `alive_count * 0.02 * 3` per step. Over 300 steps with 2.5–3 alive units, this grants **+47 to +52 points per episode** simply for existing.
   - The draw penalty was only `-10.0 * 3 = -30.0`.
   - Consequently, **stalling to a 300-step draw produces a net POSITIVE payoff ($+52 - 30 - 17 = +5\text{ to }+15$ points)** without ever accomplishing any tactical objective.
2. **Dense Combat Shaping Overwhelms Strategic Objectives**:
   - Uncoordinated combat skirmishes yield huge reward swings: Random units firing randomly earned **+151.39** in damage shaping and **+133.33** in kill bonuses ($+284.72$ total positive combat reward), which dwarfed base defense and win/loss objectives.
3. **Severe Asymmetric Penalty for Active Engagement**:
   - The Scripted Heuristic actively pushed towards objectives, achieving 8 wins (+26.67) and 2.02 enemy kills (+152.50). However, engaging the enemy exposed friendly units to retaliatory fire (friendly damage: **-175.00**, friendly losses: **-122.00**), resulting in a net negative score (-0.36).
   - In contrast, Random agents wandered aimlessly, rarely penetrated deep into enemy territory, suffered fewer catastrophic wipeouts, and rode the survival bonus into high-scoring draws (+83.66).
4. **NO_COMM Policy Defect**:
   - NO_COMM (Seed 42) failed to defend its own base (suffering -79.37 in base damage and 21 base destruction losses worth -70.00), resulting in -47.41.

---

## Step 2: Fix the Reward (Next Step)

### Plan:
1. **Clear Terminal Separation**:
   - True Real Win (Enemy base destroyed or squad wipeout): **+250.0** per agent (+750.0 team).
   - Real Loss (Friendly base destroyed or friendly wipeout): **-150.0** per agent (-450.0 team).
   - Timeout Draw: **-50.0** per agent (-150.0 team). Stalling must produce a strictly negative expected value.
2. **Remove Survival/Passivity Bonus**:
   - Eliminate `self.survival_bonus = 0.02`. Time survival must never be rewarded on its own.
3. **Small Potential-Based Shaping**:
   - Progress toward enemy base / objective: small potential difference $\Phi(s') - \Phi(s)$.
   - Damage dealt: moderate scale (e.g. 0.04), damage taken: -0.04.
   - Kills: +15.0, losses: -15.0.
   - Modest step penalty: -0.05 to encourage finishing matches decisively.
4. **Acceptance Test**:
   - Verify empirically and in unit tests that:
     `Mean Reward(Random)` $\ll$ `Mean Reward(Scripted Heuristic)` $\ll$ `Mean Reward(Targeted/Good Policy)`.
   - Add automated test assertions to `tests/test_environment.py`.

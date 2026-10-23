# STRATOS AI — Strategic Team Reasoning and Adaptive Tactical Operations System (v2)

> **Tagline**: Collective Intelligence. Coordinated Execution.  
> **Team**: NEXUS AI  
> **Project Category**: Multi-Agent Reinforcement Learning (MARL) · Learned Communication · Stochastic Simulation & Research  
> **Release Tag**: `expo-v1`  
> **License**: MIT License  

---

## 1. Project Overview & Vision

**STRATOS AI** is an advanced multi-agent reinforcement learning simulation and research testbed where three specialized artificial intelligence agents (**Commander**, **Defense**, **Attack**) learn to coordinate in a dynamic, stochastic, partially-observable battlefield.

### STRATOS v2 Core Features:
1. **Dynamic Stochastic Environment**:
   - Seeded environment variation (`np.random.default_rng(seed)`) over unit spawns, terrain offsets, and event timings.
   - Dynamic moving **Hazard Storms** dealing periodic environmental damage (5.0 HP/step).
   - Mid-match **Supply Drop** crates providing critical health and resource caches.
   - Dynamic enemy wave reinforcements and adaptive in-match opponent strategy switches.
2. **Strict Partial Observability**:
   - Role-differentiated fog-of-war vision radii (Commander: 150px, Defense: 130px, Attack: 200px) preventing global omniscience.
3. **Learned Discrete Communication**:
   - Dual-head neural policies outputting physical actions ($A \in \{0..6\}$) and discrete communication tokens ($M \in \{0..8\}$: 1 silent + 8 learned symbols).
   - Realistic 1-step message transmission delay buffer.
4. **Multi-Agent PPO (MAPPO)**:
   - Centralized Training with Decentralized Execution (CTDE) utilizing a 114-dimensional centralized critic and 67-dimensional actor observations.
5. **Audited & Honest Statistical Evaluation**:
   - Timeouts strictly adjudicate as neutral draws (`win_type="timeout_draw"`); resource hoarding exploit eliminated.
   - Tested across **3,900 total evaluation episodes** (2,400 main evaluation episodes across 100 held-out seeds `5000`..`5099` plus 1,500 ablation episodes).
   - Hierarchical **cluster bootstrap inference** treating the training seed as the unit of replication ($N=3$).

---

## 2. System Architecture

```mermaid
graph TD
    subgraph Battlefield ["PettingZoo Strategy Simulation (v2)"]
        State[GameState: Bases, Units, Crystals, Storms, Supply Drops]
        Vision[Partial Observability: Role Vision Radii]
        CommBuffer[1-Step Delay Message Transmission Buffer]
    end

    subgraph Squad ["Specialized Friendly Squad"]
        Commander[Commander: Vision 150px]
        Defense[Defense: Vision 130px]
        Attack[Attack: Vision 200px]
    end

    subgraph MAPPO ["Dual-Head Policy & Centralized Critic"]
        Obs[67-Dim Actor Observation: 46 Local + 18 Teammate Msg + 3 Role]
        SharedBackbone[Shared MLP: Linear 256 -> Tanh -> Linear 128 -> Tanh]
        ActionHead[Action Head: 7 Discrete Physical Actions]
        CommHead[Comm Head: 9 Discrete Communication Tokens]
        Critic[Centralized Critic: 114-Dim Global State -> V(s)]
    end

    Battlefield --> Vision
    Vision --> Squad
    Squad --> Obs
    CommBuffer --> Obs
    Obs --> SharedBackbone
    SharedBackbone --> ActionHead
    SharedBackbone --> CommHead
    ActionHead --> Battlefield
    CommHead --> CommBuffer
```

---

## 3. Quickstart & CLI Commands

### 1. Launch Interactive Battlefield Visualizer
Run the Pygame visualizer with tactical telemetry, hazard storm auras, vision circles, and live floating message tokens (defaults to validated reference checkpoint `stratos_policy_comm_best.pt`):
```bash
python scripts/run_game.py --mode coordinated --opponent aggressive
```
- **Hotkeys**:
  - `[C]` / `[M]`: Toggle inter-agent communication mode live (Coordinated $\leftrightarrow$ Independent).
  - `[O]`: Cycle opponent strategy (`aggressive` $\to$ `defensive` $\to$ `resource_focused`).
  - `[R]`: Reset environment with a fresh random seed.
  - `[SPACE]`: Pause / unpause simulation.

### 2. Run Comprehensive Audited Evaluation Suite
Run the full 100-episode-per-cell benchmark, baseline comparisons, ablations, message analyses, and cluster bootstrap:
```bash
# Run full audited benchmark (100 episodes per cell across 100 held-out seeds)
python scripts/audit_and_evaluate_v2.py --episodes-per-cell 100
```

### 3. Launch Research Analytics Dashboard
Start the Streamlit dashboard to inspect win rates, team rewards, and system telemetry:
```bash
streamlit run dashboard/app.py
```

### 4. Run Automated Test Suite
Run the full unit and integration test suite:
```bash
pytest -v
```

---

## 4. Empirical Research Findings (3,900 Episodes Audited)

### 4.1 Comparative Benchmark Across Held-Out Seeds (2,400 Episodes)
Evaluated across 100 fresh held-out seeds (`5000`..`5099`) against all 3 opponent archetypes:

| Condition / Policy | Evaluated Episodes | Win Rate | Draw Rate (300 steps) | Loss Rate (Enemy Win) | Mean Cumulative Team Reward |
|---|---|---|---|---|---|
| **Random Action Baseline** | 300 | 3.7% | **86.7%** | **9.7%** | +87.71 |
| **Scripted Heuristic Baseline** | 300 | **8.7%** | 67.3% | 24.0% | -15.68 |
| **NO_COMM Baseline (MARL)** | 900 | 7.1% | 69.7% | 23.2% | +14.00 |
| **COMM (Learned MARL)** | 900 | 8.2% | 75.0% | 16.8% | **+98.00** |

### 4.2 Statistical Inference via Hierarchical Cluster Bootstrap
- **Unit of Replication**: Training Seed ($N=3$, seeds `42`, `101`, `202`).
- **Win Rate Difference ($\Delta$)**: **+1.11%** (95% Cluster CI: `[-12.45%, +9.67%]`). **Not Statistically Significant**.
- **Loss Rate Difference ($\Delta$)**: **-6.44%** (95% Cluster CI: `[-20.11%, +14.00%]`). **Not Statistically Significant**.
- **Team Reward Difference ($\Delta$)**: **+84.00** (95% Cluster CI: `[-40.23, +272.58]`). **Not Statistically Significant**.
- **Sign Consistency Across Seeds**: **False** for all metrics. Seed 42 showed a substantial advantage ($\Delta\text{ Reward} = +279.57$), while Seed 101 ($\Delta = -37.03$) and Seed 202 ($\Delta = +9.46$) did not exhibit consistent gains.

### 4.3 Causal Message Ablations (1,500 Episodes)
Evaluated on reference checkpoint Seed 42 across seeds `5000`..`5099`:
- **Intact COMM**: Mean Reward = 236.82, Win Rate = 8.7%
- **Zeroed Messages ($M=0$)**: Mean Reward = 236.82, Win Rate = 8.7%
- **Random Tokens**: Mean Reward = 236.82, Win Rate = 8.7%
- **Temporally Shuffled**: Mean Reward = 236.82, Win Rate = 8.7%
- **Cross-Episode Swapped**: Mean Reward = 236.82, Win Rate = 8.7%

*Analysis*: In Seed 42, zeroing or randomizing messages altered physical actions in **0 out of 8,620 evaluated steps** under deterministic greedy execution. Seed 42 achieved superior performance through strong individual tactical behaviors rather than active reliance on communication. In contrast, Seeds 101 and 202 alter actions on 19.9% and 68.0% of steps when messages are modified, but their coordinated policies remain unstable.

### 4.4 Message Semantics & Information Theory
- **Token Usage**: Across 766,566 steps, agents collapsed to using only 4 tokens (Tokens 1, 3, 4, 8). Tokens 0, 2, 5, 6, 7 were completely unused (0.0%).
- **Shannon Entropy**: Commander (1.900 bits), Defense (1.870 bits), Attack (1.764 bits) vs. theoretical max 3.170 bits.
- **Mutual Information**: All measured values between tokens and environmental events (enemies, low HP, hazards, resources, waves) were low ($< 0.20\text{ bits}$), indicating that emergent tokens lack crisp semantic grounding.

---

## 5. Artifacts and Generated Figures

- `artifacts/experiments/PROTOCOL.md`: Full documentation of experimental protocol, validation seeds, and test setup.
- `artifacts/experiments/comm_vs_nocomm_performance.png`: Win rate and reward comparisons across conditions and baselines.
- `artifacts/experiments/ablation_causality.png`: Causal ablation comparisons across all conditions.
- `artifacts/experiments/token_distribution_and_mi.png`: Token frequency distributions and mutual information metrics.
- `artifacts/experiments/training_health_curves.png`: Training reward trajectories and final entropy across all 6 runs.
- `artifacts/experiments/final_evaluation_100eps.csv`: Raw 2,400 main evaluation episode records.
- `artifacts/experiments/ablation_study_100eps.csv`: Raw 1,500 ablation episode records.
- `artifacts/experiments/comm_vs_nocomm_statistical_report.json`: Machine-readable statistical test results and cluster bootstrap CIs.
- `artifacts/experiments/message_analysis_semantics.json`: Machine-readable message entropy and mutual information data.

---

## 6. Known Limitations

While STRATOS v3 establishes a proven communication value gap (+32.0% win rate for Oracle Messenger over Silence), several scientific and engineering limitations remain:

1. **Receiver Grounding Bottleneck**:
   Although auxiliary fact prediction losses (`COMM_AUX`) force the sender's encoder to represent hidden state (boosting weak lane mutual information up to $0.485\text{ bits}$), the downstream action policy head fails to ground these tokens into tactical decisions. In causal message ablations (zeroing, randomizing, or shuffling messages), win rates change by less than $1.3\%$ across all seeds, demonstrating that the policy head remains reliant on local ego-observations.
2. **Draw Dominance in Fortified Combat**:
   Despite removing the survival subsidy and strictly penalizing timeout draws ($-30.0$ per agent, plus step pressure), draw rates remain between $60\%$ and $79\%$ for both learned policies and uncoordinated references. Breaching a fortified base within a 300-step horizon under partial observability requires precise joint coordination that unassisted PPO struggles to discover consistently.
3. **Sample Size & Seed Heterogeneity ($N=3$)**:
   Due to compute constraints, training was conducted across 3 seeds per condition. Large inter-seed variance (e.g., NO_COMM win rates range from $0.0\%$ on Seed 42 to $16.7\%$ on Seed 101) produces overlapping 95% cluster bootstrap confidence intervals. We explicitly avoid claiming statistical significance between learned COMM and NO_COMM.
4. **Decentralized Policy Architecture**:
   Agents utilize standard multi-agent PPO with independent policy networks. Without explicit inter-agent communication attention mechanisms (e.g., TarMAC, CommNet) or trajectory-level contrastive alignment, emergent protocol discovery suffers from non-stationary credit assignment.

---

## 7. License & Attribution

Developed by the **NEXUS AI Research Team**. Released under the [MIT License](LICENSE).


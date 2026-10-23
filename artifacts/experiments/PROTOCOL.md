# STRATOS AI (v2) — Experimental Protocol & Statistical Methodology

This document outlines the exact experimental protocol, validation strategy, held-out evaluation setup, baseline controls, and statistical inference procedures used for all reported benchmarks on the `dynamic-env-v2` branch.

---

## 1. Training Protocol

### 1.1 Training Parameters & Budget
- **Algorithm**: Multi-Agent Proximal Policy Optimization (MAPPO) with Centralized Critic (114-dim global observation) and Decentralized Actors (67-dim local observation).
- **Steps Trained per Run**: 20,000 environment steps (66–67 episodes of up to 300 steps each).
- **Hardware**: AMD Ryzen 12-Core Processor (CPU-only execution, PyTorch 2.6.0+cpu).
- **Throughput**: ~285 environment steps/second during parallel rollouts.
- **Wall-Clock Time per Run**: ~70 seconds per 20,000-step training session.
- **Total Training Runs**: 6 independent runs across 2 conditions $\times$ 3 training seeds.
- **Training Seeds**: `42`, `101`, `202`.

### 1.2 Experimental Conditions
1. **Condition A: NO_COMM (Baseline)**
   - Observation space: 67 dimensions (message channels 46:64 zeroed out).
   - Single policy head: Discrete(7) physical actions.
   - Coordination mode: `independent`.
   - Checkpoints: `stratos_policy_nocomm_seed{42,101,202}.pt`.
2. **Condition B: COMM (Learned Communication)**
   - Observation space: 67 dimensions (46 local + 18 teammate one-hot tokens + 3 role one-hot).
   - Dual policy heads: Action head Discrete(7) + Communication head Discrete(9) (1 silent + 8 discrete tokens).
   - Transmission delay: 1-step message delay buffer (`self._pending_messages` delivered next step).
   - Coordination mode: `coordinated`.
   - Checkpoints: `stratos_policy_comm_seed{42,101,202}.pt`.

---

## 2. Leakage-Free Validation & Checkpoint Selection

To prevent data leakage, checkpoint selection and final testing strictly use disjoint, non-overlapping environment seeds.

### 2.1 Validation Protocol
- **Validation Environment Seeds**: `1000` to `1019` (20 distinct seeds).
- **Opponent Strategies**: All 3 opponent archetypes (`aggressive`, `defensive`, `resource_focused`).
- **Episodes per Checkpoint**: 20 seeds $\times$ 3 opponents = 60 validation episodes per training seed.
- **Selection Criterion**: The checkpoint exhibiting the highest mean cumulative team reward across the 60 validation episodes is designated as the primary reference checkpoint (`stratos_policy_comm_best.pt`).
- **Validation Results**:
  - `Seed 42`: Mean Reward = **+220.35**, Win Rate = 6.7% $\to$ **Selected as Best Checkpoint**
  - `Seed 101`: Mean Reward = **+76.35**, Win Rate = 16.7%
  - `Seed 202`: Mean Reward = **+86.33**, Win Rate = 10.0%

---

## 3. Final Held-Out Evaluation Protocol

### 3.1 Fresh Held-Out Seed Bank
- **Test Seeds**: `5000` to `5099` (100 fresh, unseeded environment runs).
- **Zero Overlap**: Neither training nor validation ever encountered seeds 5000+.

### 3.2 Evaluation Cells & Sample Size
- **Cell Definition**: `(Condition x Opponent Strategy x Training Seed)`.
- **Opponents**: 3 archetypes (`aggressive`, `defensive`, `resource_focused`).
- **Sample Allocation**:
  - `COMM`: 3 training seeds $\times$ 3 opponents $\times$ 100 seeds = **900 episodes**.
  - `NO_COMM`: 3 training seeds $\times$ 3 opponents $\times$ 100 seeds = **900 episodes**.
  - **Paired Design**: For every episode $k \in [5000, 5099]$, COMM and NO_COMM execute on the exact same initial state, storm trajectories, spawn offsets, and dynamic events.
  - `Random Baseline`: Uniform random actions $\{0..6\}$ across 3 opponents $\times$ 100 seeds = **300 episodes**.
  - `Scripted Heuristic Baseline`: Autonomous rule-based logic across 3 opponents $\times$ 100 seeds = **300 episodes**.
- **Total Main Evaluation Budget**: **2,400 episodes**.

---

## 4. Causal Ablation Suite

All ablations are evaluated across the exact same 100 held-out test seeds (`5000`..`5099`) against all 3 opponents (300 episodes per ablation condition = 1,500 total ablation episodes):

1. **Intact COMM**: Default dual-head communication with 1-step transmission delay.
2. **Zeroed Messages**: All incoming messages forced to token 0 (silent).
3. **Random Tokens**: All incoming messages replaced with uniform random integers $\sim U\{0..8\}$.
4. **Temporally Shuffled**: Incoming messages randomly sampled from previously emitted messages within the same episode.
5. **Cross-Episode Swapped**: Episode $k$ receives the entire step-by-step message sequence emitted by another episode $k'$ under the same opponent.

### Reconciliation of Previous Discrepancy:
In early preliminary iterations, the ablation study evaluated only Seed 42 (which achieved ~260 reward), whereas the headline COMM performance was averaged across all three seeds (42, 101, 202; achieving ~110 reward). In this audited protocol, all ablations are explicitly referenced against both the single-seed intact performance and the cross-seed average.

---

## 5. Statistical Inference & Replication Methodology

### 5.1 Unit of Replication
- The **unit of replication is the TRAINING SEED**, NOT the individual episode.
- Treating individual episodes as independent samples ($N=900$) creates pseudoreplication and artificially deflates standard errors.
- We report:
  1. Per-training-seed mean differences ($\Delta = \text{COMM}_s - \text{NO\_COMM}_s$) for win rate, loss rate, and reward.
  2. Sign consistency across seeds (whether all 3 seeds show positive or negative delta).
  3. **Hierarchical Cluster Bootstrap** (2,000 resamples):
     - Resamples training seeds $\{42, 101, 202\}$ with replacement.
     - Resamples paired evaluation episodes within each sampled seed.
     - Computes 95% bootstrap confidence intervals for win rate delta, loss rate delta, and reward delta.
  4. Separate significance determinations for win rate, loss rate, and team reward.

---

## 6. Message Semantics & Information-Theoretic Metrics

During the 900 COMM test episodes, every single agent step records:
- `agent_id`: Role identity (`commander`, `defense`, `attack`).
- `token`: Emitted discrete communication symbol ($0..8$).
- `enemy_visible`: Binary flag indicating whether any hostile unit is within the agent's vision cone.
- `low_hp`: Binary flag indicating unit HP $< 40\%$.
- `hazard_near`: Binary flag indicating active Hazard Storm within vision or $< 120\text{px}$.
- `resource_found`: Binary flag indicating active crystal node in vision.
- `wave_incoming`: Binary flag indicating active enemy reinforcement wave.

### Metrics Computed:
- **Token Frequency Distribution**: Empirical probability $p(m)$ for each token per agent role.
- **Shannon Entropy**: $H(M) = -\sum_{m=0}^8 p(m) \log_2 p(m)$ compared against maximum theoretical entropy $\log_2(9) \approx 3.170\text{ bits}$.
- **Mutual Information**:
  $$I(M; E) = \sum_{m \in M} \sum_{e \in E} p(m, e) \log_2 \frac{p(m, e)}{p(m)p(e)}$$
  measuring empirical correlation between message tokens and specific environmental triggers.

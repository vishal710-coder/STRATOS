# STRATOS AI — Expo Presentation Slide Deck (6 Slides)

---

## Slide 1: Title & The Scientific Audit

### **STRATOS v3: Auditing Emergent Communication in Multi-Agent RL**
*From Statistical Artifact to a Proven Communication Value Gap*

- **The Problem**: Do multi-agent reinforcement learning squads actually learn meaningful communication, or do standard benchmarks reward passive artifacts?
- **The Initial Claim**: Original report claimed learned communication boosted win rates by $>40\%$.
- **The Reality**: A rigorous reproducibility audit proved the claim was built on statistical illusions.
- **The Core Contribution**:
  - Identified and eliminated perverse reward subsidies favoring passivity.
  - Re-architected the task with asymmetric hidden facts ($40\times$ combat contrast).
  - Established a verified **$+32.0\%$ Oracle communication value gap**.
  - Transparently benchmarked learned PPO policies against ablation controls.

---

## Slide 2: The Audit: Uncovering the Artifacts

### **Why Random Policies Beat Trained Models in v1/v2**

1. **Survival Bonus Subsidized Idling**:
   - Environment granted $+0.02$ per agent/step for staying alive ($+50$ pts per episode).
   - Draw timeout penalty was only $-30.0$.
   - Stalling to a 300-step draw yielded a net *positive* score.
   - **Result**: Random policies scored $+83.66$ with $88.9\%$ draws, beating trained models ($-47.41$).
2. **Communication was Functionally Unnecessary**:
   - All agents observed identical global features; local vision was sufficient for combat.
   - Zeroing delivered messages changed performance by $0.0\%$.
   - Mutual information between messages and events was $< 0.05$ bits.
3. **Statistical Fragility**:
   - Results varied widely across 3 seeds without held-out validation.

---

## Slide 3: The Redesign: Making Communication Essential

### **Architecting Asymmetric Hidden Information in STRATOS v3**

- **Incentive Alignment**:
  - Eliminated survival subsidy ($0.0$ pts).
  - Penalized stalling draws ($-30.0$ per agent) + active step pressure ($-0.05$).
  - Added potential base progress shaping ($\Delta \Phi \times 0.5$, cycle-invariant).
- **Asymmetric Hidden Facts**:
  1. **Vulnerable Weak Lane ($L^* \in \{0, 1, 2\}$)**:
     - Uniformly randomized at reset.
     - Deals **$4.0\times$ bonus damage**; non-weak lanes take **$0.1\times$ armored damage** ($40\times$ contrast ratio).
     - Observed **ONLY** by the Attack Scout (`obs[46:49]`).
  2. **Reinforcement Flank Alert ($F^* \in \{0, 1, 2\}$)**:
     - 30-step early warning sensor of incoming sneak wave.
     - Observed **ONLY** by the Defense Guard (`obs[49:52]`).
  3. **Commander Blindness**:
     - Commander sees macro HP but is blind to $L^*$ and $F^*$. Squad coordination requires communication.

---

## Slide 4: Empirical Proof: The Oracle vs. Silence Gap

### **Proving That Communication Decides Matches (Held-Out Seeds 5000–5049)**

| Benchmark Policy | Win Rate (%) | Draw Rate (%) | Loss Rate (%) | Mean Reward [95% CI] |
|---|---|---|---|---|
| **Oracle Messenger** (Communicating) | **54.67%** | 32.67% | 12.67% | **+132.23** [103.60, 160.87] |
| **Scripted Heuristic** (Unprivileged) | **45.33%** | 33.33% | 21.33% | **+65.87** [37.98, 93.75] |
| **Silent Reference** (Messages = 0) | **22.67%** | 61.33% | 16.00% | **+8.00** [-15.49, 31.49] |
| **Random Policy** (Baseline Floor) | **6.00%** | 84.00% | 10.00% | **-23.21** [-36.86, -9.56] |

- **Communication Value Gap**:
  - **$+32.00\%$ Win Rate surge** ($2.41\times$ win rate multiplier).
  - **$+124.23$ Mean Episode Reward increase**.
  - In Silence, units stall in draws ($61.33\%$) because they attack shielded lanes and miss flanks.

---

## Slide 5: The Learned Reality: Where Neural Agents Stand

### **Do Learned PPO Agents Use Messages? (600,000 Steps, N=3 Seeds)**

| Condition (600k Steps) | Win % [95% CI] | Draw % [95% CI] | Mean Reward [95% CI] | Symbol Grounding $I(M; L^*)$ |
|---|---|---|---|---|
| **NO_COMM** (Independent) | 6.89% [0.0%, 16.7%] | 78.89% [66.7%, 91.3%] | -42.50 [-101.3, -2.5] | N/A (Blocked) |
| **COMM** (Coordinated PPO) | 6.00% [3.3%, 8.7%] | 78.67% [66.0%, 90.0%] | -36.12 [-77.5, -6.6] | 0.036 bits |
| **COMM_AUX** ($\lambda_{\text{aux}}=0.1$) | 4.67% [0.0%, 9.3%] | 74.89% [61.3%, 84.0%] | -54.65 [-79.4, -29.4] | **0.307 bits** (up to 0.485) |

- **Message Ablation Test (Intact vs Zeroed vs Random vs Shuffled)**:
  - Zeroing messages changes win rates by $< 1.3\%$ across all seeds.
  - **Verdict**: Senders learn to encode latent facts under auxiliary loss, but receivers do not yet condition actions on incoming tokens. Learned agents remain reliant on local vision.

---

## Slide 6: Takeaways & The Future of MARL Communication

### **Core Lessons for Empirical Multi-Agent AI**

1. **Mandatory Baseline Hygiene**:
   - Always evaluate against identical-architecture silence baselines on held-out seeds.
   - Decompose reward components to detect passivity subsidies.
2. **Information Asymmetry is Non-Negotiable**:
   - Emergent communication cannot be demonstrated if agents share global vision.
3. **The Groundability Gap**:
   - High mutual information in latent representations does not imply behavioral grounding in policy actions.
4. **Next Steps for STRATOS**:
   - Centralized Training with Decentralized Execution (CTDE) communication attention (TarMAC / CommNet).
   - Trajectory-level contrastive communication losses.

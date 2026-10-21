# STRATOS AI — Internal Project Progress Review Master Guide
> **Document Purpose**: End-to-End Technical & Conceptual Dossier for Project Progress Review  
> **Target Audience**: Review Tutors, Internal Evaluators, and Presenting Team Members  
> **Project Name**: STRATOS AI (*Strategic Team Reasoning and Adaptive Tactical Operations System*)  
> **Team**: NEXUS AI  
> **Domain**: Multi-Agent Reinforcement Learning (MARL) · Cooperative Tactical Systems · Simulation & Game AI  

---

## ⚡ 1. The 60-Second "Elevator Pitch" (Read This Before You Walk In!)

> **If your tutor asks: *"In simple words, what is your project about and what have you achieved so far?"***

### Your 60-Second Answer:
> *"Our project is **STRATOS AI**, an autonomous multi-agent reinforcement learning system designed to solve a fundamental challenge in artificial intelligence: **How do specialized AI agents cooperate when facing complex adversarial threats?**
>
> We built a real-time, top-down strategy battlefield in Python using the **PettingZoo Parallel API** and **Pygame**. On this battlefield, a squad of three specialized friendly agents—a **Commander**, a **Defense Guard (Tank)**, and an **Attack Striker (Scout)**—must protect their headquarters, gather crystal resources, and defeat enemy squads.
>
> The core research question we investigated is: **Does real-time communication between agents measurably improve their performance compared to agents operating independently?**
>
> To answer this scientifically, we implemented an Actor-Critic policy trained via **Proximal Policy Optimization (PPO)** coupled with a dedicated **Tactical Communication Bus**. We then conducted a strict controlled ablation experiment across **150 episodes over 5 random seeds**.
>
> **Our Key Finding**: Coordinated agents achieved a **100% win rate** compared to **66.7%** for independent agents—a **+33.3 percentage point gain**. Most dramatically, against aggressive rush attacks, independent agents suffered total squad wipeouts (**0% win rate**), whereas coordinated agents communicated emergency defense directives to organize a unified perimeter, achieving a **100% win rate**.
>
> All 20 unit and integration tests are passing, and we have an interactive visualizer and a live Streamlit analytics dashboard to demonstrate today."*

---

## 🎮 2. Game Concept & Battlefield Mechanics

### 2.1 The Gameplay Premise
STRATOS AI is modeled after real-time strategy (RTS) games like *StarCraft* or *Age of Empires*, but stripped down to a pure mathematical research testbed:
- **Battlefield Arena**: Continuous 2D arena ($800 \times 600$ pixels) with bounded borders.
- **Friendly Headquarters**: Located at $(80, 300)$ with **1,000 HP**. If destroyed, friendly loses.
- **Enemy Headquarters**: Located at $(720, 300)$ with **1,000 HP**. If destroyed, friendly wins.
- **Crystal Depots (Resources)**: Three mining nodes situated in contested territory:
  - North Depot: $(400, 150)$
  - Center Depot: $(400, 300)$
  - South Depot: $(400, 450)$
  Crystals can be mined by moving onto depots. Resources fuel team score and victory calculations.

### 2.2 Friendly Squad Specialization (Asymmetric Roles)
Unlike generic multi-agent systems where all units are clones, STRATOS employs **heterogeneous role specialization**:

| Unit Role | Max HP | Move Speed | Attack Damage | Attack Range | Vision Range | Role Description |
|---|---|---|---|---|---|---|
| **Commander** | 280 HP | 3.2 px/s | 18.0 HP | 95.0 px | 220.0 px | Balanced macro strategist; manages mining priorities & issues strategic fallback orders. |
| **Defense Guard** | 380 HP | 2.6 px/s | 22.0 HP | 85.0 px | 190.0 px | Heavy armored tank; slow moving, anchors base perimeter, absorbs fire and intercepts intruders. |
| **Attack Striker** | 220 HP | 3.8 px/s | 32.0 HP | 120.0 px | 240.0 px | Agile glass-cannon; high damage, long firing range, patrols and flanks enemy assets. |

### 2.3 Adversary Profiles (The Three Opponent Strategies)
To test how well the AI adapts, we created three distinct scripted benchmark opponents:
1. **Aggressive Rush**: Disregards crystals and immediately sprints towards the friendly base with all units to execute a rapid blitzkrieg.
2. **Defensive Turtle**: Camps tightly inside the enemy base perimeter ($150\text{px}$ radius), refusing to engage unless friendly units enter their territory.
3. **Resource-Focused**: Greedily mines central crystal nodes until hoarding 100 crystals, then transitions into a coordinated late-game siege.

### 2.4 Match Resolution (Win/Loss Rules)
A match terminates upon any of three conditions:
1. **Base Annihilation**: A base reaches $0\text{ HP}$ (Instant Win / Loss).
2. **Squad Wipeout**: All 3 combat units of a team are eliminated (Instant Loss for that team).
3. **Time Limit (300 Steps)**: If time expires, the team with the highest combined score wins:
   $$\text{Score} = \text{Base HP} + (0.5 \times \text{Mined Resources})$$

---

## 🛠️ 3. Complete Technical Stack

| Layer | Technology | Purpose & Role in STRATOS |
|---|---|---|
| **Programming Language** | **Python 3.11+** | Core runtime for simulation, neural nets, and tooling. |
| **Deep Learning Framework** | **PyTorch (`torch`)** | Neural network definition, Actor-Critic computation, autograd, and GPU/CPU tensor ops. |
| **Simulation Standard** | **PettingZoo (Parallel API) & Gymnasium** | The recognized industry standard for multi-agent reinforcement learning environments. |
| **RL Algorithm** | **Proximal Policy Optimization (PPO)** | Actor-Critic on-policy optimization with Generalized Advantage Estimation (GAE-$\lambda$). |
| **Numerical Processing** | **NumPy** | Vectorized vector mathematics, observation normalization, and distance matrices. |
| **Visual Simulation** | **Pygame** | 60-FPS interactive tactical canvas, HUD telemetry, laser firing FX, and live communication ticker. |
| **Analytics & Dashboard** | **Streamlit, Pandas, Matplotlib** | Web-based research UI for exploring reward curves, win rates, and seed distributions. |
| **Configuration** | **PyYAML** | Centralized game parameters, hyperparameters, and environment settings (`config/game.yaml`). |
| **Automated Testing** | **Pytest** | 20 unit and integration tests verifying kinematics, message decay, and policy sensitivities. |

---

## 🏛️ 4. End-to-End System Architecture

```mermaid
graph TD
    subgraph Environment ["Simulation Layer (PettingZoo 800x600)"]
        State[GameState: Bases, Units, Crystals, Cooldowns]
        Combat[Combat Engine: Range Checking & Attack Cooldowns]
        Harvest[Resource Engine: Crystal Extraction]
    end

    subgraph CommBus ["Tactical Communication Layer"]
        Router[Tactical Coordinator & Mailboxes]
        Decay[Temporal Freshness Filter: Age <= 15 steps]
        Encoder[6-Channel Comm Feature Extractor]
    end

    subgraph NeuralPolicy ["Multi-Agent Neural Network (PyTorch)"]
        Obs[Normalized 49-Dimensional Vector]
        MLP[Shared 2-Layer Trunk: 128 Hidden Units + LayerNorm + Tanh]
        ActorHead[Actor Head: Linear(128, 7) -> Categorical Logits]
        CriticHead[Critic Head: Linear(128, 1) -> State Value V(s)]
    end

    subgraph Agents ["Executing Units"]
        CA[Commander Unit]
        DA[Defense Guard Unit]
        AA[Attack Striker Unit]
    end

    State --> Obs
    CommBus --> Encoder
    Encoder --> Obs
    Obs --> MLP
    MLP --> ActorHead & CriticHead
    ActorHead --> CA & DA & AA
    CA & DA & AA --> Router
    Router --> Decay
    Decay --> CommBus
    CA & DA & AA --> Combat & Harvest
    Combat & Harvest --> State
```

### 4.1 The Observation Space ($D = 49$ Dimensions)
Every step, each agent receives a continuous vector normalized to $[-1.0, 1.0]$:

| Channels | Category | Size | What the Agent "Sees" |
|---|---|---|---|
| `0:5` | **Self State** | 5 | Normalized $(x, y)$, current HP ratio, attack cooldown ratio, alive flag (1.0 or 0.0). |
| `5:8` | **Friendly HQ** | 3 | Normalized position $(x, y)$ and remaining HP ratio. |
| `8:11` | **Enemy HQ** | 3 | Normalized position $(x, y)$ and remaining HP ratio. |
| `11:19` | **Squad Teammates** | 8 | For both teammates: relative $(\Delta x, \Delta y)$, HP ratio, alive flag. |
| `19:31` | **Visible Enemies** | 12 | For all 3 enemy units: relative $(\Delta x, \Delta y)$, HP ratio, alive flag. |
| `31:40` | **Resource Nodes** | 9 | For all 3 crystal deposits: relative $(\Delta x, \Delta y)$, remaining crystal ratio. |
| `40:43` | **Global Metrics** | 3 | Friendly team resources, enemy resources, match step progress ratio ($t / t_{max}$). |
| `43:49` | **Tactical Comm Bus** | **6** | Active directive code, threat level, target $(\Delta x, \Delta y)$, support flag, priority. |

*(Note: In optional role-conditioned policies, channels `49:52` append a 3-dimensional one-hot vector identifying `[Commander, Defense, Attack]`)*.

### 4.2 The Action Space (7 Discrete Tactical Primitives)
Rather than raw low-level steering physics (which drastically inflates RL sample complexity), agents select from **7 tactical macro-actions**:
0. `HOLD`: Hold current position and conserve formation.
1. `MOVE_BASE`: Fall back to friendly base perimeter to defend.
2. `MOVE_ENEMY_BASE`: Advance toward enemy headquarters to lay siege.
3. `MOVE_RESOURCE`: Navigate to the nearest active crystal depot to mine.
4. `ATTACK_NEAREST_ENEMY`: Engage and fire upon the closest hostile combat unit.
5. `DEFEND_PERIMETER`: Patrol the forward defensive perimeter ($120\text{px}$ forward buffer of HQ).
6. `TACTICAL_RETREAT`: Disengage and maneuver away from concentrated enemy fire.

---

## 📡 5. The Inter-Agent Communication Bus

### 5.1 Why Symbolic Communication?
Many AI researchers attempt "continuous latent communication" (where neural networks pass raw mathematical vectors). We chose a **Structured Symbolic Bus** for three vital engineering reasons:
1. **100% Interpretability**: Human evaluators and tutors can read the messages in real time (`ENEMY_DETECTED`, `DEFEND_BASE`). There are no black-box mysteries.
2. **Bandwidth Efficiency**: Low computational and transmission overhead.
3. **Safety & Human-in-the-Loop**: A human commander can inspect or override messages.

### 5.2 Supported Message Directives
- `ENEMY_DETECTED` (Priority 0.9): Fired when enemies cross into friendly territory ($x < 440.0$).
- `DEFEND_BASE` (Priority 0.8): Emergency order commanding units to break off mining and protect HQ.
- `REQUEST_SUPPORT` (Priority 0.7): Sent automatically when an agent's HP drops below 40%.
- `ATTACK_TARGET` (Priority 0.6): Synchronized offensive ping targeting vulnerable enemy units.
- `RESOURCE_LOW` (Priority 0.3): Scarcity notification when team crystal reserves fall below 50.
- `RESUME_OFFENSE` (Priority 0.5): Broadcast when threat level subsides, authorizing offensive advance.

### 5.3 The 15-Step Temporal Freshness Decay
In early experiments, we discovered a subtle defect: when an agent asked for support, that message stayed in the mailbox forever, locking teammates into defensive mode long after the enemy died.  
We engineered a **Temporal Freshness Filter**:
$$\text{Age}(m) = t_{\text{current}} - t_{\text{dispatch}}$$
Messages with $\text{Age}(m) > 15 \text{ steps}$ are automatically expired and dropped from observation encoding.

---

## 🧠 6. Reinforcement Learning: PPO Algorithm Details

### 6.1 Why PPO (Proximal Policy Optimization)?
Standard Policy Gradient algorithms (like REINFORCE) often take destructive gradient steps that ruin a trained policy. **PPO** solves this using a **clipped surrogate objective**, ensuring that policy updates stay within a trusted region ($\epsilon = 0.2$):

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio between the new and old policy.
- $\hat{A}_t$ is the Generalized Advantage Estimation (GAE-$\lambda$) measuring whether an action performed better than expected.
- $\epsilon = 0.2$ prevents the policy from deviating excessively in a single update.

### 6.2 Total Loss Function
$$L_{total} = L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t)$$
- $L^{VF}(\theta) = \frac{1}{2} (V_\theta(s_t) - R_t)^2$ is the Critic Mean Squared Value Error ($c_1 = 0.5$).
- $S[\pi_\theta]$ is the Entropy bonus ($c_2 = 0.01$), which encourages exploration and prevents early policy collapse.

### 6.3 Reward Function Design (`environment/rewards.py`)
Agents receive multi-faceted rewards:
- **Combat Success**: $+0.2$ per point of damage dealt to enemies; $+20.0$ for destroying an enemy unit.
- **Base Protection**: $+50.0$ for winning; $-50.0$ for losing; penalty for friendly base taking damage.
- **Resource Extraction**: $+0.1$ per crystal mined.
- **Perimeter Defense Incentive**: A targeted $+0.04$ bonus for the Defense Guard when holding perimeter ($x < 350$) while an invasion is underway ($x < 440$).

---

## 🔬 7. The Scientific Proof: Multi-Seed Controlled Ablation Study

### 7.1 What is an "Ablation Study"?
To scientifically prove that *communication* was the true cause of victory (and not just luck, random chance, or unit stats), we conducted a strict **ablation experiment**:
- **Coordinated Mode**: Policy receives full 49-dimensional observations, including the 6 communication channels (43–48).
- **Independent Baseline**: Policy receives the exact same physical observations (0–42), but communication channels (43–48) are **strictly clamped to zero**.
- **Fairness Guarantee**: Both modes run on the **identical neural network checkpoint**, across **identical random seeds**, facing **identical initial unit placements**.

### 7.2 The 150-Episode Benchmark Matrix (5 Seeds: 42, 101, 202, 303, 404)

| Scenario / Opponent | Mode | Win Rate (Wins / Total) | Mean Team Reward | Survival Rate | Mean Messages Exchanged |
|---|---|---|---|---|---|
| **Aggressive Rush** | **Independent** | **0 / 25 (0.0%)** | -408.89 | 0.0% (Wipeout) | 0.0 msgs |
| **Aggressive Rush** | **Coordinated** | **25 / 25 (100.0%)** | **+242.26** | **33.3%** | **819.0 msgs** |
| **Defensive Turtle** | Independent | 25 / 25 (100.0%) | +556.86 | 100.0% | 0.0 msgs |
| **Defensive Turtle** | Coordinated | 25 / 25 (100.0%) | +556.86 | 100.0% | 697.0 msgs |
| **Resource Focused** | Independent | 25 / 25 (100.0%) | +408.82 | 33.3% | 0.0 msgs |
| **Resource Focused** | Coordinated | 25 / 25 (100.0%) | +408.82 | 33.3% | 503.0 msgs |

### 7.3 Aggregate Summary Across All 150 Episodes:

| Metric | Independent Baseline (75 Episodes) | Coordinated System (75 Episodes) | Net Coordination Advantage |
|---|---|---|---|
| **Overall Win Rate** | 50 / 75 (**66.7%**) | 75 / 75 (**100.0%**) | **+33.3 percentage points** |
| **Mean Cumulative Reward** | +185.60 points | **+402.65 points** | **+217.05 points** |
| **Aggressive Rush Win Rate** | **0.0% (Total Squad Wipeout)** | **100.0% (Flawless Victory)** | **+100.0 percentage points** |
| **Average Transmissions** | 0.0 messages | **673.0 messages** | Active real-time coordination |

### 7.4 Why Did Independent Agents Fail Against the Aggressive Rush?
When the aggressive enemy rushes at step 0:
- **Independent Agents**: They have no inter-agent awareness. The Commander and Attack Striker wander forward toward crystal nodes. The enemy squad ambushes them 3-on-1, wipes them out, and then destroys the Defense Guard. **Result: 0% survival, 0% win rate.**
- **Coordinated Agents**: The moment an enemy is sighted, an emergency `DEFEND_BASE` directive flashes across the communication bus. The entire squad instantly halts outward movement, falls back into a tight defensive triangle around the HQ, and uses concentrated crossfire to wipe out the invaders. **Result: 100% win rate.**

---

## 💻 8. Live Demonstration Playbook (Step-by-Step for Tomorrow)

### 8.1 Launching the Interactive Pygame Visualizer
Open a terminal in the project directory (`e:\STRATOS`) and execute:

```powershell
.\.venv\Scripts\python scripts/run_demo.py
```
*(Optional: add `--opponent aggressive`, `--opponent defensive`, or `--opponent resource_focused`)*

### 8.2 Interactive Keyboard Controls to Show the Tutors:
- **`[M]` (The Showstopper Key)**: **Toggles live between Coordinated and Independent mode!**
  - Press `[M]`: The bottom radio ticker instantly goes silent (`0 messages`). Watch how agents disperse.
  - Press `[M]` again: Returns to Coordinated mode. The radio ticker lights up green again!
- **`[SPACE]`**: Pause or resume the simulation.
- **`[UP / DOWN]`**: Speed up or slow down the simulation FPS (great if tutors want to examine combat slow-motion).
- **`[R]`**: Reset episode with a fresh seed.
- **`[Q / ESC]`**: Close the window.

### 8.3 What to Point Out on the Screen During the Demo:
1. **Top HUD Bar**: Friendly base HP (blue), Enemy base HP (red), mined crystal count, and active macro directive.
2. **Combat Visualization**: Colored unit circles (Commander = Gold, Defense = Shield Blue, Attack = Red/Orange), firing laser lines, and health bars above each unit.
3. **Bottom Communication Ticker**: Shows live radio messages streaming across the bus in real time (e.g., `[STEP 42] DEFENSE -> ALL: ENEMY_DETECTED (P:5)`).
4. **End-of-Match Victory Card**: Shows final outcome, surviving units, total damage dealt, and total messages exchanged.

### 8.4 Launching the Research Analytics Dashboard:
Open a second terminal and execute:
```powershell
.\.venv\Scripts\streamlit run dashboard/app.py
```
This opens an interactive web dashboard in the browser showing:
- Interactive Win Rate Bar Charts (Independent vs. Coordinated).
- Cumulative Reward Distribution curves.
- Scenario-by-scenario multi-seed breakdown tables.

---

## 🛠️ 9. Real Engineering Obstacles Overcome (Demonstrates Authenticity!)

Tutors love hearing about real technical challenges you encountered and how you solved them:

1. **The Stale Mailbox Defect**:
   - *Problem*: Early on, agents kept retreating to base forever. We discovered that messages in `inboxes` had no expiration date, so a single distress call from step 20 lingered at step 200.
   - *Solution*: Built the **15-Step Temporal Decay Filter** in `coordinator.py`.
2. **Broken Command Succession**:
   - *Problem*: If the Commander unit died, the coordinator never updated directives because of `if commander_unit.is_alive:`. Directives locked permanently.
   - *Solution*: Implemented dynamic command succession—if Commander falls, leadership shifts to Defense Guard, then Attack Striker.
3. **Circular Import Bug**:
   - *Problem*: `TacticalCoordinator` importing `GameState` and `GameState` importing `TacticalCoordinator` broke test execution.
   - *Solution*: Decoupled coordinator imports and added typing annotations (`state: Any`).

---

## 🛡️ 10. Tutor Q&A Defense Matrix (The Top 10 Review Questions)

### Q1: *"Did you just hardcode rules for the agents, or is this genuine Reinforcement Learning?"*
> **Answer**:  
> *"It is genuine Reinforcement Learning. The agents' actions are generated by an **Actor-Critic Neural Network** trained via **PPO** in PyTorch. The policy receives a 49-dimensional continuous observation vector and outputs probability distributions over 7 discrete actions. What is structured is the **communication schema** (`MessageType` and priority), which feeds directly into observation channels 43 to 48. The policy network itself learned when to heed those messages through gradient optimization."*

### Q2: *"How do you prove the neural policy actually uses the communication channels rather than ignoring them?"*
> **Answer**:  
> *"We conducted a formal **Policy Sensitivity Audit** on the neural network weights. By keeping physical observations (channels 0–42) completely identical and only modifying the communication channels (43–48), we observed:
> 1. A measurable shift in output action logits ($\Delta_{max} = 0.0516$).
> 2. A substantial shift in the Critic's state-value estimation ($\Delta V(s) = +0.931$).
> 3. Under stochastic sampling, identical initial seeds produced 8 distinct action divergence events within a single episode.
> Furthermore, in our 150-episode benchmark, zeroing out the comm channels caused a 100% loss against aggressive rushers, while active comm channels delivered a 100% win rate."*

### Q3: *"Why did you use PettingZoo instead of standard Gymnasium?"*
> **Answer**:  
> *"Gymnasium is built for single-agent systems (`env.step(action)`). PettingZoo is the multi-agent standard, specifically the **Parallel API** (`env.step({agent_1: a1, agent_2: a2, ...})`), which enables simultaneous agent execution, decentralized observations, and individualized reward vectors."*

### Q4: *"Why PPO over DQN or DDPG/SAC?"*
> **Answer**:  
> *"DQN struggles with non-stationary multi-agent dynamics because the replay buffer contains transitions collected under earlier policies of other agents. SAC/DDPG are continuous action space algorithms, whereas our macro-actions are discrete. PPO with Generalized Advantage Estimation is on-policy, stable, resilient to hyperparameter swings, and prevents destructive policy updates via surrogate clipping."*

### Q5: *"Why did you use 7 macro-actions instead of continuous $(v_x, v_y)$ velocity control?"*
> **Answer**:  
> *"Continuous low-level kinematic control requires millions of exploratory steps just to learn how to walk in a straight line, which obscures our research question. By abstracting kinematics into 7 tactical action primitives (e.g., `MOVE_BASE`, `ATTACK_NEAREST_ENEMY`), the reinforcement learning algorithm can focus its sample complexity on high-level strategic reasoning and coordination."*

### Q6: *"Why did both independent and coordinated modes achieve 100% win rate against the Defensive Turtle?"*
> **Answer**:  
> *"Because defensive turtle opponents camp inside their base and never threaten our headquarters. Even without communication, individual agents eventually stumble upon crystal nodes or slowly advance to siege the enemy base. Communication shines most when coordinating under acute, time-sensitive adversarial pressure like an aggressive rush."*

### Q7: *"What is the size and architecture of your neural network?"*
> **Answer**:  
> *"It consists of an observation input of 49 dimensions, a shared representation trunk of two hidden linear layers with 128 units each, LayerNorm, and Tanh activation functions with orthogonal weight initialization. It then branches into two heads: an **Actor Head** (Linear $128 \rightarrow 7$) outputting categorical action distributions, and a **Critic Head** (Linear $128 \rightarrow 1$) estimating the scalar state value $V(s)$."*

### Q8: *"How do you prevent communication channels from suffering from message flooding?"*
> **Answer**:  
> *"Three mechanisms: First, messages have priority ratings (1 to 5). Second, the Tactical Coordinator maintains inboxes and only extracts the single highest-priority active message per agent. Third, we enforce a strict 15-step temporal freshness decay filter that discards expired messages."*

### Q9: *"How many tests have you written, and how do you ensure code quality?"*
> **Answer**:  
> *"We have a 20-test automated regression suite written with Pytest. It tests agent kinematics, combat resolution, message freshness decay, communication-to-policy dataflow, and training checkpoint resumption. All 20 tests pass at 100% in under 12 seconds."*

### Q10: *"What are the current limitations and your next sprint roadmap?"*
> **Answer**:  
> *"Our current system uses a 2D point-mass kinematic model and symbolic communication. In our next sprint, we plan to:
> 1. Scale training to 30,000+ timesteps to further compress policy entropy.
> 2. Introduce **Asymmetric Self-Play**, where friendly squads train against evolving opponent AI squads rather than scripted opponents.
> 3. Experiment with **continuous latent communication** (such as TarMAC or CommNet) to compare emergent neural protocols against our symbolic protocol."*

---

## 📋 11. Quick Pre-Presentation Checklist (Morning of Review)

- [ ] Open PowerShell in `e:\STRATOS`
- [ ] Activate virtual environment: `.\.venv\Scripts\activate`
- [ ] Verify test suite passes: `pytest -q` (Should show `20 passed`)
- [ ] Verify Pygame demo launches: `python scripts/run_demo.py`
- [ ] Test the `[M]` key during demo to ensure you can toggle Coordinated / Independent mode smoothly
- [ ] (Optional) Launch Streamlit dashboard in a browser tab: `streamlit run dashboard/app.py`
- [ ] Keep this guide open on your phone or tablet for quick reference!

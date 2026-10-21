# STRATOS AI — Project Presentation & Demonstration Guide
## Slide-by-Slide Outline, Speaking Script & Demonstration Walkthrough

**Project**: STRATOS AI (Strategic Team Reasoning and Adaptive Tactical Operations System)  
**Team**: NEXUS AI Research Laboratory  
**Presenter Guide Duration**: ~7–10 Minutes  

---

## Slide 1: Title & Introduction (1 Minute)

### Slide Visual:
* **Title**: STRATOS AI — Collective Intelligence. Coordinated Execution.
* **Subtitle**: Multi-Agent Reinforcement Learning and Inter-Agent Communication in Real-Time Strategic Combat.
* **Team**: NEXUS AI
* **Key Graphic**: Hero screenshot of the battlefield visualizer showing the friendly squad and communication bus.

### Presenter Script:
> *"Good morning/afternoon everyone. Today, I am proud to present **STRATOS AI**, an autonomous multi-agent reinforcement learning system that investigates a fundamental question in artificial intelligence:*  
>  
> ***'Does real-time communication between specialized AI agents measurably improve their coordination and tactical performance compared to agents operating independently?'***  
>  
> *In real-time multi-agent environments—from autonomous drone swarms to robotic defense—individual agents often suffer from coordination breakdown. When agents can't talk to each other, they make localized, selfish decisions that lead to catastrophic failures. In STRATOS AI, we created a complete simulation testbed, designed a real-time tactical communication bus, and benchmarked the results across 150 independent trials."*

---

## Slide 2: Problem Statement & Research Methodology (1.5 Minutes)

### Slide Visual:
* **Core Research Question**: Coordinated Multi-Agent Squad vs. Independent Baseline.
* **Controlled Ablation Graphic**:
  * Physical observations (Channels 0–42) $\rightarrow$ Identical for both.
  * Communication channels (Channels 43–48) $\rightarrow$ Active in Coordinated; Strictly Clamped to Zero in Independent.
* **Key Opponent Strategies**:
  1. *Aggressive Rush*: Frontal assault directly against the base.
  2. *Defensive Turtle*: Fortified base perimeter defense.
  3. *Resource-Focused*: Economic crystal harvesting before transitioning to an assault.

### Presenter Script:
> *"To make this a scientifically rigorous study, we avoided guesswork by applying strict ablation testing:*  
>  
> *Both our **Coordinated squad** and our **Independent baseline squad** use the exact same neural policy weights, start on identical random seeds, and face identical unit placements.*  
>  
> *The only difference: in Independent mode, the communication channels are strictly zeroed out. In Coordinated mode, the agents share real-time alerts like `ENEMY_DETECTED`, `DEFEND_BASE`, and `REQUEST_SUPPORT`.*  
>  
> *We then pitted both configurations against three distinct opponent tactical behaviors: Aggressive Rushers, Defensive Turtlers, and Resource Harvesters."*

---

## Slide 3: System Architecture & Agent Roles (1.5 Minutes)

### Slide Visual:
* **Battlefield Map Graphic**: 800 $\times$ 600 arena with Bases, Crystal Depots, and Combat Range Circles.
* **Agent Squad Breakdown**:
  * **Commander**: Balanced HP & speed; coordinates macro directives and crystal mining.
  * **Defense Guard**: 380 HP tank; slow-moving interceptor that shields the HQ base.
  * **Attack Striker**: High damage (32 HP/strike) agile scout with long attack range.
* **Neural Pipeline**: PettingZoo Parallel API $\rightarrow$ 49/52-Dim Observation $\rightarrow$ Actor-Critic Policy (PPO) $\rightarrow$ 7 Discrete Macro-Actions.

### Presenter Script:
> *"The battlefield simulation is built according to PettingZoo parallel environment standards. Our friendly squad consists of three specialized units:*  
> - *The **Commander**, who sets strategic priorities,*  
> - *The **Defense Guard**, a high-armor tank that intercepts invaders,*  
> - *And the **Attack Striker**, a fast, high-damage unit specialized in long-range engagements.*  
>  
> *At every step, each unit receives a normalized observation vector and passes it through an Actor-Critic neural network trained using Proximal Policy Optimization (PPO). The policy outputs one of 7 macro-actions, including holding ground, perimeter defense, enemy engagement, or crystal harvesting."*

---

## Slide 4: Empirical Findings & Results (2 Minutes)

### Slide Visual:
* **The 150-Episode Benchmark Comparison Table**:
  * Overall Win Rate: **Independent = 66.7%** vs **Coordinated = 100.0%** (**+33.3% Communication Delta**).
  * Aggressive Scenario: **Independent = 0.0%** vs **Coordinated = 100.0%** (**+100.0% Improvement**).
  * Mean Cumulative Team Reward: **+185.60** vs **+402.65** (**+217.05 Point Advantage**).
  * Average Messages: **673 Transmissions per Coordinated Episode**.
* **Key Chart**: Side-by-side win-rate comparison bar chart (from the Streamlit dashboard).

### Presenter Script:
> *"Here are our key empirical findings across 150 benchmark episodes evaluated over 5 distinct random seeds:*  
>  
> *1. **A Clear Win-Rate Advantage**: Coordinated agents achieved a perfect **100.0% win rate** across all scenarios, compared to **66.7%** for the independent baseline—a statistically verified **+33.3 percentage point advantage**.*  
>  
> *2. **The Aggressive Rush Breakthrough**: When facing immediate aggressive base rushes, independent agents suffered a 100% squad wipeout loss (0.0% win rate). Why? Because individual agents wandered off toward crystal nodes and were picked off one by one.*  
>  
> *In contrast, coordinated agents broadcasted emergency `DEFEND_BASE` radio directives immediately upon sighting hostiles. The entire squad retreated to form a tight defensive perimeter around the headquarters, completely neutralizing the enemy wave and winning 100% of matches.*  
>  
> *3. **Communication Volume**: Coordinated agents exchanged an average of 673 tactical messages per match, verifying that our communication pipeline is active and driving real tactical decisions."*

---

## Slide 5: Live Demonstration (2 Minutes)

### Actions to Perform:
1. Open PowerShell and run:
   ```powershell
   .\.venv\Scripts\python.exe scripts/run_demo.py
   ```
2. Point out:
   * The **Top HUD Bar**: shows Friendly HP, Enemy HP, crystal counts, and active strategic directive.
   * The **Units in Combat**: Defense Guard anchoring base, Commander and Attack contesting midfield.
   * The **Bottom Communication Bus Ticker**: streaming live transmissions like `[DEFENSE -> COMMANDER: ENEMY_DETECTED]` in real time.
3. Demonstrate live interactive toggling:
   * Press **`[M]`** on the keyboard to switch to **INDEPENDENT MODE**.
   * Note that the ticker goes silent (`0 messages`), communication features zero out, and show how agents disperse.
   * Press **`[M]`** again to return to **COORDINATED MODE**.
4. Show the **End-of-Match Victory Overlay**:
   * Point out the match conclusion reason, total damage dealt, surviving units, and total messages exchanged.

### Presenter Script:
> *"Now let's see the system live in action.*  
> *(Launch `run_demo.py`)*  
> *On screen, you can see our battlefield. Notice the bottom ticker: this is our real-time tactical communication bus. As soon as the defense unit detects enemy units crossing midfield, it immediately sends high-priority radio alerts to the squad.*  
>  
> *Notice what happens when I press the `[M]` key: we toggle into Independent Mode. The radio stream goes silent, communication channels are clamped to zero, and the units can no longer coordinate their defense.*  
>  
> *When the match concludes, our tactical card shows the complete post-match telemetry—final headquarters health, surviving units, crystal mining totals, and total radio transmissions."*

---

## Slide 6: Research Analytics Dashboard & Conclusion (1 Minute)

### Slide Visual:
* Quick view of the Streamlit dashboard running in the browser (`streamlit run dashboard/app.py`).
* Summary bullet points:
  * PettingZoo RL Simulation standard.
  * PPO Actor-Critic architecture with 15-step message freshness decay.
  * Verified +33.3% communication win-rate advantage.
  * 20/20 automated unit & integration tests passing.

### Presenter Script:
> *"In addition to the visual simulator, we built an interactive Streamlit research dashboard that lets researchers and evaluators inspect training dynamics, reward learning curves, and scenario-by-scenario distributions in real time.*  
>  
> *All code is fully tested with a 20-test automated test suite passing at 100%.*  
>  
> *In conclusion, STRATOS AI proves that structured, low-bandwidth communication channels prevent catastrophic failures and unlock significant, measurable performance gains in multi-agent reinforcement learning.*  
>  
> *Thank you, and I would now be happy to take any questions."*

---

## Slide 7: Potential Evaluator Q&A Preparation

### Q1: *"How do you prove the agents are actually using the messages, rather than just reacting to seeing the enemy?"*
* **Answer**: *"We performed controlled policy sensitivity audits where we fed the exact same physical coordinates into the network and only altered the 6 communication channels. We observed a measurable shift in action logits ($\Delta_{max} = 0.0516$) and a $+0.931$ shift in critic state-value estimation. Furthermore, in our 150-episode multi-seed benchmark, independent agents facing identical enemy positions lost 100% of matches against aggressive rushers, while coordinated agents won 100%."*

### Q2: *"Why did you use symbolic messages instead of continuous latent vectors (like CommNet)?"*
* **Answer**: *"Symbolic messaging allows for full interpretability and debugging. Evaluators and human observers can read exactly what the AI is communicating (`DEFEND_BASE`, `REQUEST_SUPPORT`) in real time, making the system transparent and safe for human-in-the-loop applications."*

### Q3: *"What is the purpose of the 15-step freshness filter?"*
* **Answer**: *"Early in development, we found that without message expiration, emergency support requests stayed in agent inboxes forever, causing agents to stay in defensive fallback loops even after the enemy was already eliminated. The 15-step filter ensures that tactical directives expire once transient threats pass."*

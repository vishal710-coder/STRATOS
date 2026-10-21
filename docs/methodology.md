# STRATOS AI — Research Methodology & Experimental Design

## 1. Research Hypothesis

**Core Research Question**:
> *"Does communication between specialized AI agents improve their coordination and performance against different opponent strategies compared with agents that operate independently?"*

### Null Hypothesis ($H_0$)
Inter-agent communication provides no statistically measurable improvement in task completion, win rate, or team survival relative to non-communicating independent agents ($p > 0.05$).

### Alternative Hypothesis ($H_1$)
Structured communication significantly improves team survivability and coordinated response efficiency against aggressive and resource-focused opponent behaviors.

---

## 2. Experimental Design Matrix

The evaluation suite executes a full factorial matrix:

$$\text{Configurations} = \{\text{Independent Baseline}, \text{Coordinated Multi-Agent}\} \times \{\text{Aggressive}, \text{Defensive}, \text{Resource-Focused}\}$$

1. **Scenario 1 — Baseline vs Aggressive**: Independent agents without communication facing direct assault.
2. **Scenario 2 — Coordinated vs Aggressive**: Coordinated agents exchanging threat alerts and defensive fallback directives.
3. **Scenario 3 — Baseline vs Defensive**: Independent agents against turtled, perimeter-holding opponents.
4. **Scenario 4 — Coordinated vs Defensive**: Coordinated squad facing turtled opponents.
5. **Scenario 5 — Baseline vs Resource-Focused**: Independent agents facing opponents prioritizing crystal harvesting.
6. **Scenario 6 — Coordinated vs Resource-Focused**: Coordinated agents facing resource-focused opponents.

---

## 3. Evaluation Protocol & Metrics

To prevent data contamination, evaluation is strictly separated from training:
- **Seed Control**: Fixed evaluation seed ($12345$) across scenarios ensuring identical initial entity placements.
- **Sample Size**: Configurable evaluation episodes per scenario (default $15-20$ episodes).
- **Recorded Metrics**:
  - **Win Rate**: $\frac{N_{wins}}{N_{total}}$
  - **Loss Rate**: $\frac{N_{losses}}{N_{total}}$
  - **Draw / Timeout Rate**: $\frac{N_{timeouts}}{N_{total}}$
  - **Friendly Squad Survival Rate**: Mean proportion of friendly combatants surviving at episode termination.
  - **Mean Cumulative Team Reward**: Aggregate discounted reward earned across the episode.
  - **Mean Episode Duration**: Number of discrete time steps until terminal resolution.
  - **Communication Density**: Total verified message transmissions routed across the squad.

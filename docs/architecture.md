# STRATOS AI — System Architecture Specification

## 1. System Overview

STRATOS AI (**Strategic Team Reasoning and Adaptive Tactical Operations System**) is a multi-agent reinforcement learning (MARL) research prototype engineered to investigate how inter-agent communication and tactical coordination influence squad-level outcomes against divergent opponent strategies.

```mermaid
graph TD
    subgraph Environment ["PettingZoo Strategy Simulation Environment"]
        State[GameState: Bases, Units, Resources, Timers]
        Combat[Combat Engine: Range, Cooldowns, Damage]
        Harvest[Resource Engine: Nodes, Stockpile]
    end

    subgraph Communication ["Tactical Message Bus"]
        Bus[Coordinator & Router]
        Mailbox[Agent Mailboxes: Commander, Defense, Attack]
    end

    subgraph Agents ["Specialized Friendly Agents"]
        CA[Commander Agent: Strategy & Priority]
        DA[Defense Agent: Perimeter & Base Protection]
        AA[Attack Agent: Target Assault & Flanking]
    end

    subgraph Policy ["Actor-Critic PPO Policy"]
        Encoder[Observation Encoder: 49 Dimensions]
        Actor[Actor Network: 7 Discrete Action Primitives]
        Critic[Critic Network: Value V(s) Estimation]
    end

    subgraph Opponent ["Configurable Opponent Squad"]
        OppAI[Aggressive / Defensive / Resource-Focused Strategy]
    end

    State --> Encoder
    Bus --> Encoder
    Encoder --> Policy
    Actor --> CA & DA & AA
    CA & DA & AA --> State
    CA & DA & AA --> Bus
    Bus --> Mailbox
    OppAI --> State
    State --> Combat & Harvest
```

---

## 2. Observation Space Specification

Each agent receives a dense, normalized continuous observation vector of dimension **$D = 49$** bounded within $[-1.0, 1.0]$:

| Index Range | Category | Dimension | Normalized Content |
|---|---|---|---|
| `0:5` | **Self Status** | 5 | $(x_{norm}, y_{norm}, \frac{hp}{hp_{max}}, \frac{cd}{cd_{max}}, is\_alive)$ |
| `5:8` | **Friendly HQ** | 3 | $(x_{norm}, y_{norm}, \frac{hp}{hp_{max}})$ |
| `8:11` | **Enemy HQ** | 3 | $(x_{norm}, y_{norm}, \frac{hp}{hp_{max}})$ |
| `11:19` | **Squad Teammates** | 8 | 2 other agents $\times (rel\_x, rel\_y, hp_{ratio}, is\_alive)$ |
| `19:31` | **Visible Enemies** | 12 | 3 enemy units $\times (rel\_x, rel\_y, hp_{ratio}, is\_alive)$ |
| `31:40` | **Resource Nodes** | 9 | 3 nodes $\times (rel\_x, rel\_y, amount_{ratio})$ |
| `40:43` | **Global Metrics** | 3 | $(res_{friendly}, res_{enemy}, \frac{step}{max\_steps})$ |
| `43:49` | **Communication Bus** | 6 | $(directive, threat\_level, target\_dx, target\_dy, support\_flag, priority)$ |

---

## 3. Action Space Specification

Each agent selects from **7 discrete tactical action primitives**:

1. `HOLD (0)`: Remain stationary; maintain tactical stance.
2. `MOVE_BASE (1)`: Fall back to perimeter of friendly headquarters.
3. `MOVE_ENEMY_BASE (2)`: Advance directly toward enemy headquarters to lay siege.
4. `MOVE_RESOURCE (3)`: Navigate to nearest active crystal deposit to harvest.
5. `ATTACK_NEAREST_ENEMY (4)`: Maneuver into firing range and engage closest hostile unit.
6. `DEFEND_PERIMETER (5)`: Patrol defensive perimeter ($120\text{px}$ forward buffer of HQ).
7. `TACTICAL_RETREAT (6)`: Fall back away from concentrated hostile fire.

---

## 4. Multi-Agent Communication Architecture

The communication system operates in two testable modalities:

1. **Coordinated Mode**:
   - Structured messages (`ENEMY_DETECTED`, `REQUEST_SUPPORT`, `DEFEND_BASE`, `RESUME_OFFENSE`) are dispatched to the `TacticalCoordinator`.
   - The coordinator validates, routes to targeted inboxes, and encodes state into 6 continuous observation features for the neural policy.
2. **Independent Mode (Baseline)**:
   - Communication channels are suppressed (zeroed out).
   - Agents act purely on localized sensory observations without inter-agent signaling.

---

## 5. Reinforcement Learning Implementation

- **Algorithm**: Proximal Policy Optimization (PPO) with clipped surrogate objective ($\epsilon = 0.2$).
- **Advantage Estimation**: Generalized Advantage Estimation (GAE-$\lambda$) with $\gamma = 0.99, \lambda = 0.95$.
- **Actor-Critic Backbone**:
  - Input: $49$ dimensions.
  - Shared trunk: Two hidden layers of $128$ units each with orthogonal initialization and $\tanh$ activations.
  - Actor Head: Categorical distribution over $7$ actions.
  - Critic Head: Scalar state-value $V(s)$.
- **Loss Function**:
  $$\mathcal{L}_{PPO}(\theta) = \hat{\mathbb{E}}_t \left[ \mathcal{L}_t^{CLIP}(\theta) - c_1 \mathcal{L}_t^{VF}(\theta) + c_2 \mathcal{S}[\pi_\theta](s_t) \right]$$
  where $c_1 = 0.5$ (value loss coefficient) and $c_2 = 0.01$ (entropy bonus coefficient).

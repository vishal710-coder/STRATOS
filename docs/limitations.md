# STRATOS AI — Limitations and Technical Constraints

In adherence to strict research integrity and engineering honesty, this document catalogs the established limitations, hardware envelope boundaries, and scope constraints of the STRATOS AI research prototype.

---

## 1. Physical Environment & Kinematics Approximations

- **Kinematics**: Units employ 2D point-mass kinematics with clamped vector movement rather than rigid-body physics engines (such as Box2D or PhysX). Collisions are handled via distance checks and map boundary clamping rather than complex impulse momentum transfer.
- **Line of Sight & Fog of War**: The observation vector provides relative spatial offsets to entities within vision range ($190-240\text{px}$). Obstacles and occlusion masks (e.g., raytraced vision blocking) are simplified to preserve high-throughput step rates on CPU hardware.

---

## 2. Communication Topology

- **Predefined Symbolic Protocol**: The primary prototype implements structured symbolic messages (`ENEMY_DETECTED`, `REQUEST_SUPPORT`, `DEFEND_BASE`, `RESUME_OFFENSE`) mapped to observation feature vectors rather than emergent continuous latent communication vectors (e.g., CommNet or TarMAC).
- **Synchronous Bus**: Messages are delivered to recipient inboxes within the same simulation step and processed synchronously rather than modeling asynchronous network latency or packet drops.

---

## 3. Policy Parameter Sharing

- **Shared Actor-Critic Backbone**: The current release utilizes a shared multi-agent neural policy across the three roles, conditioned upon role identities and specific communication features. While practical and stable for fast training on laptop hardware, fully decentralized specialized networks trained via MADDPG or MAPPO with distinct actor heads represent a future scalability horizon.

---

## 4. Hardware Envelope & Training Scale

- **Compute Envelope**: Optimized for execution on local laptops (Ryzen 5 / RTX 3050 Laptop GPU / 16GB RAM). Training experiments are designed for $6,000 - 30,000$ timesteps, providing fast convergence for university demonstrations rather than billion-step large-scale esports clusters (e.g., AlphaStar or OpenAI Five).
- **Opponent Policies**: Opponent strategies are rule-based scripted baselines (Aggressive rush, Defensive turtle, Resource hoarder). Opponent policies do not concurrently train via self-play in this release.

"""STRATOS AI - Streamlit Research Analytics Dashboard."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Configure Page
st.set_page_config(
    page_title="STRATOS AI // Research Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main { background-color: #0b0f19; color: #e2e8f0; }
    .stMetric { background-color: #151d30; padding: 14px; border-radius: 8px; border: 1px solid #2d3b5e; }
    h1, h2, h3 { color: #38bdf8 !important; font-family: 'Consolas', monospace; }
    .badge-coord { background-color: #0369a1; color: white; padding: 4px 8px; border-radius: 4px; }
    .badge-indep { background-color: #475569; color: white; padding: 4px 8px; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚡ STRATOS AI — Multi-Agent Research Analytics")
st.caption("Strategic Team Reasoning and Adaptive Tactical Operations System | NEXUS AI Research Laboratory")

# Sidebar navigation and checkpoint selector
st.sidebar.header("🕹️ Experiment Controls")
checkpoints_dir = Path("artifacts/checkpoints")
experiments_dir = Path("artifacts/experiments")

checkpoints = list(checkpoints_dir.glob("*.pt")) if checkpoints_dir.exists() else []
if checkpoints:
    selected_ckpt = st.sidebar.selectbox("Active Checkpoint", [str(c.name) for c in checkpoints])
    st.sidebar.success(f"Selected: {selected_ckpt}")
else:
    st.sidebar.warning("No saved checkpoints found in artifacts/checkpoints.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Core Research Question")
st.sidebar.info(
    "\"Does communication between specialized AI agents improve their coordination and performance "
    "against different opponent strategies compared with agents that operate independently?\""
)

# Data paths
history_csv = experiments_dir / "training_history.csv"
eval_csv = experiments_dir / "evaluation_results.csv"
eval_json = experiments_dir / "evaluation_summary.json"

tab_overview, tab_eval, tab_training, tab_config = st.tabs(
    ["📊 Executive Summary", "⚔️ Coordinated vs Independent Evaluation", "📈 Training Dynamics", "⚙️ System Architecture"]
)

# TAB 1: EXECUTIVE SUMMARY
with tab_overview:
    st.subheader("Experimental Results Overview")

    if eval_csv.exists() and eval_json.exists():
        df_eval = pd.read_csv(eval_csv)
        with open(eval_json, "r") as f:
            summary_data = json.load(f)

        # High-level KPIs
        coord_mask = df_eval["coordination_mode"] == "coordinated"
        indep_mask = df_eval["coordination_mode"] == "independent"

        coord_win_rate = df_eval[coord_mask]["winner"].eq("friendly").mean()
        indep_win_rate = df_eval[indep_mask]["winner"].eq("friendly").mean()
        delta_win_rate = coord_win_rate - indep_win_rate

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Coordinated Win Rate", f"{coord_win_rate:.1%}")
        col2.metric("Independent Baseline Win Rate", f"{indep_win_rate:.1%}")
        col3.metric("Communication Delta", f"{delta_win_rate:+.1%}", delta=f"{delta_win_rate:+.1%}")
        col4.metric("Total Evaluation Episodes", len(df_eval))

        st.markdown("---")

        # Win Rate Comparison Chart by Opponent Strategy
        st.subheader("Win Rate Comparison across Opponent Behaviors")
        opp_strategies = df_eval["opponent_strategy"].unique()
        chart_data = []

        for strat in opp_strategies:
            c_rate = df_eval[(df_eval["opponent_strategy"] == strat) & coord_mask]["winner"].eq("friendly").mean()
            i_rate = df_eval[(df_eval["opponent_strategy"] == strat) & indep_mask]["winner"].eq("friendly").mean()
            chart_data.append({"Opponent Strategy": strat.capitalize(), "Coordinated": c_rate * 100.0, "Independent": i_rate * 100.0})

        df_chart = pd.DataFrame(chart_data).set_index("Opponent Strategy")
        st.bar_chart(df_chart, color=["#38bdf8", "#94a3b8"])

    else:
        st.info("No evaluation data recorded yet. Run `python scripts/evaluate_model.py` to generate research benchmarks.")

# TAB 2: DETAILED EVALUATION
with tab_eval:
    st.subheader("Scenario-by-Scenario Evaluation Breakdown")

    if eval_csv.exists():
        df_eval = pd.read_csv(eval_csv)

        # Filter by opponent
        strat_filter = st.selectbox("Filter Opponent Strategy", ["All"] + list(df_eval["opponent_strategy"].unique()))
        if strat_filter != "All":
            filtered_df = df_eval[df_eval["opponent_strategy"] == strat_filter]
        else:
            filtered_df = df_eval

        st.dataframe(
            filtered_df[[
                c for c in [
                    "episode_id", "scenario_name", "coordination_mode", "opponent_strategy",
                    "winner", "win_type", "duration_steps", "team_reward", "friendly_units_survived",
                    "enemy_units_eliminated", "friendly_resources_collected", "messages_exchanged"
                ] if c in filtered_df.columns
            ]],
            use_container_width=True,
        )

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Team Reward Distribution")
            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0b0f19")
            ax.set_facecolor("#151d30")
            for mode, color in [("coordinated", "#38bdf8"), ("independent", "#f43f5e")]:
                subset = filtered_df[filtered_df["coordination_mode"] == mode]
                if not subset.empty:
                    ax.hist(subset["team_reward"], alpha=0.6, label=mode.capitalize(), color=color, bins=12)
            ax.set_xlabel("Cumulative Team Reward", color="#e2e8f0")
            ax.set_ylabel("Frequency", color="#e2e8f0")
            ax.tick_params(colors="#94a3b8")
            ax.legend()
            st.pyplot(fig)

        with c2:
            st.markdown("#### Friendly Unit Survival")
            fig2, ax2 = plt.subplots(figsize=(6, 3.5), facecolor="#0b0f19")
            ax2.set_facecolor("#151d30")
            modes = ["coordinated", "independent"]
            surv_means = [filtered_df[filtered_df["coordination_mode"] == m]["friendly_units_survived"].mean() for m in modes if m in filtered_df["coordination_mode"].values]
            if surv_means:
                ax2.bar([m.capitalize() for m in modes if m in filtered_df["coordination_mode"].values], surv_means, color=["#38bdf8", "#94a3b8"])
            ax2.set_ylabel("Avg Units Survived (Max 3)", color="#e2e8f0")
            ax2.tick_params(colors="#94a3b8")
            st.pyplot(fig2)

    else:
        st.warning("Run evaluation experiments via `python scripts/evaluate_model.py`.")

# TAB 3: TRAINING DYNAMICS
with tab_training:
    st.subheader("PPO Policy Training History")

    if history_csv.exists():
        df_hist = pd.read_csv(history_csv)

        colA, colB, colC = st.columns(3)
        colA.metric("Total Training Steps", df_hist["step"].max())
        colB.metric("Total Episodes Completed", df_hist["episode"].max())
        colC.metric("Peak Training Win Rate", f"{df_hist['friendly_win_rate'].max():.1%}")

        st.markdown("#### Reward Learning Curve")
        st.line_chart(df_hist.set_index("episode")["episode_reward"], color="#38bdf8")

        st.markdown("#### Rolling Win Rate over Training")
        st.line_chart(df_hist.set_index("episode")["friendly_win_rate"], color="#22c55e")
    else:
        st.info("No training history found. Launch training via `python scripts/train_model.py` to record learning dynamics.")

# TAB 4: SYSTEM ARCHITECTURE
with tab_config:
    st.subheader("System Architecture & Configuration Specifications (v2)")
    st.markdown(
        """
        - **MARL Paradigm**: MAPPO (Multi-Agent PPO) with Dual-Head Actor (Shared Weights) and Centralized Critic
        - **Observation Space**: 67 Dimensions (Partially Observable):
            - 46 dims: Local state (unit status, friendly/enemy base, visible teammates, visible enemies, resources, global, storm)
            - 18 dims: Received learned message tokens (2 teammates × 9 one-hot tokens, 1-step transmission delay)
            - 3 dims: Agent role one-hot badge (`commander`, `defense`, `attack`)
        - **Action Space**:
            - Kinematic head: 7 discrete primitives (Hold, Move Base, Move Enemy Base, Move Resource, Attack Nearest Enemy, Defend Perimeter, Tactical Retreat)
            - Communication head: 9 discrete learned tokens (0=silent, 1..8=learned tactical signals)
        - **Stochastic Dynamics & Events**:
            - Seeded RNG (`np.random.default_rng`) governing spawns, stat jitter, dynamic drift
            - Dynamic hazard storms, enemy wave spawns, opponent strategy switching, supply crates
            - Timeout adjudication strictly yields **DRAW** (no resource hoarding win)
        """
    )

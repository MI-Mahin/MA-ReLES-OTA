# MA-ReLES-OTA

### Multi-Agent Reinforcement Learning for Coordinated Automotive OTA Updates

> A high-performance research extension of the **ReLES-OTA** framework — scaling single-ECU firmware optimization into a full **Multi-Agent RL** coordination system for resource-constrained vehicular networks.

---

## 🔬 Updated Research Hypothesis & Focus

As modern vehicles transition into software-defined platforms, they house a heterogeneous fleet of Electronic Control Units (ECUs) with conflicting safety, latency, and reliability requirements. Our research has evolved from a simple baseline comparison to addressing a fundamental architectural question in Multi-Agent Reinforcement Learning (MARL):

### The Parameter-Sharing Dilemma in Heterogeneous Fleets
*   **Partial Parameter Sharing (FP3O)**: Keeps a shared feature extraction backbone but uses specialized policy/action heads per ECU type to prevent policy interference.
*   **Full Parameter Sharing with Type-Conditioning (IPPO/MAPPO)**: Employs a single unified network but conditions the inputs on a one-hot `ecu_type` vector, relying on the network's capacity to represent distinct behaviors internally.

### Formal Hypotheses
*   **$H_0$ (Null)**: Explicit partial parameter sharing (specialized action heads per ECU type) is required to resolve conflicting action policies (e.g., safety-critical Engine preferring multi-base delta updates vs. Infotainment preferring simple copy operations).
*   **$H_1$ (Alternative)**: Full parameter sharing with type-conditioning achieves equal or superior returns and convergence rates due to 100% sample reuse across the fleet, making specialized policy heads redundant.

---

## 📊 Key Experimental Findings

Our 2-million timestep, multi-seed benchmarks revealed critical insights for vehicular MARL scheduling:

1.  **Homogeneous Fleet Parity**: In a homogeneous environment (where ECU type labels do not alter reward functions), all algorithms converged to the same return (~12.85). IPPO/MAPPO had a slight edge in sample efficiency, as FP3O's specialized heads only received 25% of the update data.
2.  **Reward-Scale Heterogeneity Failure**: Uniformly scaling rewards (e.g., Engine paying 1.5× more than Infotainment for all actions) does not change the ordinal preference of operations. All agents still preferred the cheapest option (Copy), rendering specialization unnecessary.
3.  **Qualitative Policy Divergence (The Per-Operation Cost Model)**: We introduced operation-specific cost multipliers (Engine penalized for binary patch modification but discounted for multi-base verification; Infotainment penalized for verification). This forced a qualitative policy conflict: Engine must learn to prioritize verification, while Infotainment must prioritize copying.
4.  **The `agent_id` Confounder**: Early benchmarks showed IPPO dominating because it observed the specific `agent_id` (a unique index). The shared head used this to memorize per-agent actions, acting as a hidden heterogeneous lookup. By replacing `agent_id` with a semantic `ecu_type` one-hot vector, we established a fair comparison.
5.  **Validation of $H_1$**: Even in a fair comparison, **Type-Conditioned IPPO (`16.71`) outperformed/equaled FP3O (`15.88`)**. The shared network has sufficient capacity to resolve type conditioning, while benefit of pooling 100% of fleet transitions to train one set of weights outweighs the specialized head inductive bias.

---

## 🏗️ Architectural Pillars

| Pillar | Concept / Paper | Implementation |
| :--- | :--- | :--- |
| **Stability** | MAPPO / RSR-RSMARL | Centralized Critic with **Death Masking** (retaining `ecu_type` for terminated agents to prevent distribution shifts in vectorized environments). |
| **Safety** | CBF Safety Shield | Real-time safety filter that intercepts unsafe action choices to prevent memory budget overflows before they execute. |
| **Fairness** | MARL-CC | Monte Carlo **Shapley Value** calculation in the environment's coalition evaluation to attribute rewards based on an ECU's marginal efficiency contribution. |
| **Sample Efficiency** | Type-Conditioned IPPO | Fully shared actor-critic weights with one-hot `ecu_type` input conditioning. |

---

## 🛠️ Installation & Tests

1.  **Clone and install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Verify environment compliance**:
    ```bash
    python test_marl_env.py
    ```
3.  **Verify policy routing architecture**:
    ```bash
    python test_fp3o.py
    ```

---

## 👥 Contributors & Roles

*   **Saadman Sakib**: Lead Architect (Policy Routing, Env Design & CTDE Integration)
*   **Mohtasim Dipto**: Environment Developer (Death Masking & Parallel Vectorization)
*   **Mahin Islam**: Statistical & Analytical Lead (Welch's T-Tests, Confidence Intervals & Plot Generation)

---

<p align="center">
  <i>Undergraduate Thesis Research — Software & Systems Lab</i>
</p>

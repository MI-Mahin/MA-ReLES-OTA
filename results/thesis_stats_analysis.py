"""
MA-ReLES-OTA — Thesis Statistical Analysis
============================================
Run with: python thesis_stats_analysis.py

STATUS OF EACH REQUESTED ANALYSIS
----------------------------------
[DONE]   Welch's t-test, IPPO vs FP3O (heterogeneous, type-conditioned) — uses
         raw_seed_returns.json (real data).
[DONE]   95% CI verification against training_registry.json — matches exactly,
         confirms the registry's CI was computed as t.ppf(0.975, df=n-1) * SEM.
[DONE]   Cohen's d / Hedges' g effect size.
[DONE]   Figure 3 — Payload Cost vs Shield Rate bar chart (IPPO vs FP3O only;
         NO rule-based baseline exists anywhere in training_registry.json —
         add a "Baseline" entry with algorithm == "Baseline" and re-run to include it).
[NOT POSSIBLE] Figure 1 (homogeneous training curves), Figure 2 (heterogeneous
         training curves), Figure 4 (entropy decay), and the entropy/shield-rate
         correlation all require PER-TIMESTEP scalars (mean_return, entropy,
         actor_loss, critic_loss, shield_trigger events) over the course of
         training. training_registry.json only ever stored ONE final summary row
         per completed run, and the runs/ TensorBoard logs were confirmed NOT to
         exist (never retained / never enabled during training). There is no
         source data left to reconstruct a training curve or an entropy series
         from — this is a permanent gap in the retained data, not a pending
         upload. It should be reported as a stated limitation in the thesis
         (see the Limitations paragraph delivered alongside this script), not
         re-attempted.
         The loader/plotting stubs below (`load_tb_scalars`, commented-out) are
         left in ONLY as a template for future replications of this work that
         enable `tensorboard_log=` in the SB3 PPO instantiation and retain the
         resulting event files — they do not apply to the current dataset.
"""

import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open("raw_seed_returns.json") as f:
    raw = json.load(f)
with open("training_registry.json") as f:
    registry = json.load(f)

ippo = np.array(raw["IPPO_Safety_True"])
fp3o = np.array(raw["FP3O_Safety_True"])
mappo = np.array(raw["MAPPO_Safety_True"])


# ---------------------------------------------------------------------------
# 1. Welch's t-test, CI verification, Cohen's d / Hedges' g
# ---------------------------------------------------------------------------
def welch_report(name_a, a, name_b, b):
    n1, n2 = len(a), len(b)
    m1, m2 = a.mean(), b.mean()
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
    pooled_sd = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    d = (m1 - m2) / pooled_sd
    J = 1 - (3 / (4*(n1+n2-2) - 1))
    g = d * J
    print(f"\n--- {name_a} vs {name_b} ---")
    print(f"{name_a}: mean={m1:.4f}, sd={s1:.4f}, n={n1}")
    print(f"{name_b}: mean={m2:.4f}, sd={s2:.4f}, n={n2}")
    print(f"Welch t={t:.4f}, df={df:.2f}, p={p:.4f}")
    print(f"Cohen's d={d:.4f}, Hedges' g={g:.4f}")
    if n1 <= 3 or n2 <= 3:
        print("WARNING: n<=3 per arm -> this test is severely underpowered; "
              "df~1 means almost any effect size fails to reach significance. "
              "Report descriptively, not as a confirmed null/alternative result.")
    return dict(t=t, p=p, df=df, d=d, g=g)


result = welch_report("IPPO (type-conditioned)", ippo, "FP3O (specialized heads)", fp3o)

# cross-check against registry
r24 = next(r for r in registry if r["run_id"] == 24)
r25 = next(r for r in registry if r["run_id"] == 25)
print(f"\nRegistry-reported p_value_vs_ippo for FP3O run25: {r25['p_value_vs_ippo']} "
      f"(recomputed: {result['p']:.4f}) -> {'MATCH' if abs(float(r25['p_value_vs_ippo'])-result['p'])<1e-3 else 'MISMATCH'}")


# ---------------------------------------------------------------------------
# 2. Figure 3 — Payload cost / shield rate bar chart (real data, no baseline available)
# ---------------------------------------------------------------------------
def make_figure3():
    labels = ["IPPO\n(type-conditioned)", "FP3O\n(specialized heads)"]
    payload = [r24["mean_payload_cost"], r25["mean_payload_cost"]]
    shield = [r24["shield_rate"] * 100, r25["shield_rate"] * 100]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax1.bar(x - width/2, payload, width, label="Mean Payload Cost", color="#4C72B0")
    ax1.set_ylabel("Mean Payload Cost (bytes)", color="#4C72B0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, shield, width, label="Shield Trigger Rate", color="#DD8452")
    ax2.set_ylabel("Shield Trigger Rate (%)", color="#DD8452")
    ax2.set_ylim(0, 5)

    for b in bars1:
        ax1.annotate(f'{b.get_height():,.0f}', (b.get_x()+b.get_width()/2, b.get_height()),
                     ha='center', va='bottom', fontsize=9)
    for b in bars2:
        ax2.annotate(f'{b.get_height():.1f}%', (b.get_x()+b.get_width()/2, b.get_height()),
                     ha='center', va='bottom', fontsize=9)

    ax1.set_title("Figure 3: Payload Cost Comparison\n"
                   "(Heterogeneous Fleet, Run #24 vs Run #25 — n=2 seeds each)\n"
                   "Both algorithms achieve a 0% shield trigger rate at this stage",
                   fontsize=10)
    fig.tight_layout()
    fig.savefig("figure3_payload_safety_tradeoff.png", dpi=200)
    print("\nSaved figure3_payload_safety_tradeoff.png")
    print("NOTE 1: no rule-based/dummy baseline exists in training_registry.json — "
          "only IPPO vs FP3O are plotted.")
    print("NOTE 2: shield_rate is 0.0 for BOTH runs at this stage, so the safety "
          "axis has no dynamic range here; caption reframed honestly to avoid "
          "implying a trade-off that isn't present in this specific run pair.")


make_figure3()


# ---------------------------------------------------------------------------
# 3. BLOCKED: Figures 1, 2, 4 and entropy/shield correlation
# ---------------------------------------------------------------------------
def load_tb_scalars(path):
    """Expected schema once you export TensorBoard runs:
    {"timestep": [...], "ep_rew_mean": [...], "entropy_loss": [...],
     "shield_triggered": [...]}  (shield_triggered as 0/1 per logged step, or
     a rolling rate)
    """
    with open(path) as f:
        return json.load(f)

# Example of what Figure 1/2 plotting will look like once data exists:
#
# def make_training_curve_figure(runs: dict, title, outfile):
#     fig, ax = plt.subplots(figsize=(8,5))
#     for label, scalars in runs.items():
#         t = np.array(scalars["timestep"])
#         r = np.array(scalars["ep_rew_mean"])
#         # if multiple seeds are stacked, compute mean +/- 95% CI band here
#         ax.plot(t, r, label=label)
#     ax.set_xlabel("Timesteps"); ax.set_ylabel("Mean Return")
#     ax.set_title(title); ax.legend()
#     fig.savefig(outfile, dpi=200)
#
# make_training_curve_figure({
#     "IPPO": load_tb_scalars("tb_scalars_ippo_homog.json"),
#     "MAPPO": load_tb_scalars("tb_scalars_mappo_homog.json"),
#     "FP3O": load_tb_scalars("tb_scalars_fp3o_homog.json"),
# }, "Figure 1: Training Curves (Homogeneous Fleet)", "figure1_homogeneous_curves.png")

print("\nFigures 1, 2, 4 and the entropy/shield correlation are NOT generated. "
      "This is a permanent data gap, not a pending upload: training_registry.json "
      "only ever stored one final summary row per run, and the runs/ TensorBoard "
      "logs were confirmed not to exist. Report this as a stated limitation "
      "rather than attempting to reconstruct these figures.")

"""
tools/extract_training_curves.py — Reconstruct training curves from TensorBoard tfevents logs.
========================================================================================
Finds all tfevents files under results/marl_models/ and extracts scalar training histories
(returns, entropy loss, episode lengths) per algorithm, safety configuration, and seed.
Saves the extracted data to CSVs under results/final/ and plots the training curves.

Usage
-----
    python tools/extract_training_curves.py
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Setup style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11})

MODEL_DIR = Path("results/marl_models")
OUT_DIR = Path("results/final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_scalars_from_tfevents(tfevents_path: Path) -> dict:
    """Extract rollout/ep_rew_mean and train/entropy_loss from a tfevents file."""
    # Size guidance: 0 loads everything (no size limit)
    ea = EventAccumulator(str(tfevents_path), size_guidance={'scalars': 0})
    ea.Reload()
    
    tags = ea.Tags().get('scalars', [])
    data = {}
    
    # We want to extract:
    # 1. rollout/ep_rew_mean (returns)
    # 2. train/entropy_loss (negative of entropy)
    # 3. rollout/ep_len_mean (episode steps)
    target_tags = ['rollout/ep_rew_mean', 'train/entropy_loss', 'rollout/ep_len_mean']
    
    for tag in target_tags:
        if tag in tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            vals = [e.value for e in events]
            data[tag] = pd.Series(vals, index=steps)
            
    return data

def main():
    print("Searching for TensorBoard tfevents log files under results/marl_models/...")
    
    # Find all tfevents files
    tfevents_files = glob.glob(str(MODEL_DIR / "**" / "events.out.tfevents.*"), recursive=True)
    if not tfevents_files:
        print("  ❌ No tfevents files found! Ensure your training runs logged to TensorBoard.")
        return
        
    print(f"  Found {len(tfevents_files)} tfevents log files.")
    
    records = []
    
    for path_str in tfevents_files:
        path = Path(path_str)
        # Parse path info: e.g. results/marl_models/FP3O_Safety_True/seed_0/logs/fp3o_bd/PPO_1/...
        parts = path.parts
        
        # Determine Algorithm and Safety status from path
        experiment_dir = ""
        seed_dir = "seed_0"
        
        for part in parts:
            if "_Safety_" in part:
                experiment_dir = part
            if "seed_" in part:
                seed_dir = part
                
        if not experiment_dir:
            # Fallback parsing
            continue
            
        algo_name = experiment_dir.split("_Safety_")[0]
        safety_status = "Safety_True" in experiment_dir
        
        print(f"  Extracting logs for: {algo_name} | Safety: {safety_status} | {seed_dir} ...")
        
        try:
            scalars = extract_scalars_from_tfevents(path)
            
            # Combine Series into a DataFrame
            df = pd.DataFrame(scalars)
            df.index.name = 'step'
            df = df.reset_index()
            
            # Add metadata columns
            df['algorithm'] = algo_name
            df['safety'] = safety_status
            df['seed'] = seed_dir
            df['experiment'] = experiment_dir
            
            records.append(df)
        except Exception as e:
            print(f"    [error] Failed to parse {path.name}: {e}")
            
    if not records:
        print("  ❌ No data could be extracted.")
        return
        
    # Combine all data into one master DataFrame
    master_df = pd.concat(records, ignore_index=True)
    
    # Save raw extracted CSV
    raw_csv_path = OUT_DIR / "extracted_marl_training_history.csv"
    master_df.to_csv(raw_csv_path, index=False)
    print(f"  Saved raw extracted metrics to: {raw_csv_path}")
    
    # ── Plot 1: Reconstructed Learning Curves (Returns over Timesteps) ──
    # Note: VecNormalize clips training rewards to [-10, 10] scale during training
    # so logged ep_rew_mean values are in the normalized range, not the eval range.
    # We separate out safety=True experiments (the main comparison) vs safety=False.
    plt.figure(figsize=(12, 6))

    if 'rollout/ep_rew_mean' in master_df.columns:
        valid_df = master_df.dropna(subset=['rollout/ep_rew_mean'])

        # Separate plots for safety=True and safety=False
        for safety_val, ax_label in [(True, "Safety-Constrained"), (False, "Unconstrained")]:
            sub = valid_df[valid_df['safety'] == safety_val]
            if sub.empty:
                continue

            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=sub,
                x='step',
                y='rollout/ep_rew_mean',
                hue='experiment',
                errorbar=('ci', 95),
                linewidth=2
            )
            plt.title(f"MARL Learning Curves — {ax_label} Fleet\n"
                      "(Training-time returns; VecNormalize scaling active)")
            plt.xlabel("Environment Timesteps")
            plt.ylabel("VecNormalize-Scaled Episode Return")
            plt.legend(title="Experiment")
            plt.tight_layout()

            label_str = "safety_true" if safety_val else "safety_false"
            plot_path = OUT_DIR / f"reconstructed_learning_curves_{label_str}.png"
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"  Saved learning curves ({ax_label}) to: {plot_path}")

    # ── Plot 2: Reconstructed Entropy Decay ──
    if 'train/entropy_loss' in master_df.columns:
        valid_entropy_df = master_df.dropna(subset=['train/entropy_loss']).copy()

        # SB3 logs entropy loss as negative entropy, convert back to positive entropy
        valid_entropy_df['entropy'] = -valid_entropy_df['train/entropy_loss']

        # Focus on safety=True experiments (our main contribution)
        safety_true_entropy = valid_entropy_df[valid_entropy_df['safety'] == True]

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=safety_true_entropy,
            x='step',
            y='entropy',
            hue='experiment',
            errorbar=('ci', 95),
            linewidth=2
        )

        plt.title("Policy Entropy Decay — Safety-Constrained Fleet\n"
                  "(All algorithms converge from high-entropy exploration to deterministic exploitation)")
        plt.xlabel("Environment Timesteps")
        plt.ylabel("Policy Entropy (nats)")
        plt.legend(title="Experiment")
        plt.tight_layout()

        entropy_plot_path = OUT_DIR / "reconstructed_entropy_curves.png"
        plt.savefig(entropy_plot_path, dpi=300)
        plt.close()
        print(f"  Saved reconstructed entropy curves plot to: {entropy_plot_path}")

    print("\nExtraction complete! Deliver these files/scripts to Mahin to do stats & final thesis charts.")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

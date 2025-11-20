"""
Evaluation Script for Hybrid EA+DRL Fidelity-Aware System

Compares performance of:
1. Baseline DRLQ (time-only optimization)
2. Hybrid EA+DRL (fidelity-aware optimization)
3. Heuristic baselines (greedy, random, etc.)
"""

import json
import os
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

from env_creator import env_creator
from env_wrapper import FidelityAwareRewardWrapper


DATASET_PATH = "./qdataset/qd_all.pickle"


def evaluate_policy(
    policy_name: str,
    w_time: float = 0.7,
    w_fidelity: float = 0.3,
    num_episodes: int = 50,
) -> Dict:
    """
    Evaluate a policy configuration.

    Args:
        policy_name: Name of the policy being evaluated
        w_time: Weight for time reward
        w_fidelity: Weight for fidelity reward
        num_episodes: Number of evaluation episodes

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {policy_name}")
    print(f"Weights: w_time={w_time:.3f}, w_fidelity={w_fidelity:.3f}")
    print(f"{'='*60}\n")

    # Create environment
    env_config = {
        "dataset": DATASET_PATH,
        "evaluation": True,
        "policy": policy_name,
    }
    base_env = env_creator(env_config)
    env = FidelityAwareRewardWrapper(base_env, w_time=w_time, w_fidelity=w_fidelity)

    # Evaluation metrics
    episode_rewards = []
    total_fidelities = []
    total_times = []
    rescheduling_counts = []
    success_counts = []

    # Run episodes
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_fidelities = []
        episode_successes = 0
        episode_rescheduling = 0

        while not done:
            # Random policy for demonstration (replace with trained agent)
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward

            # Collect fidelity information
            if 'fidelity_score' in info:
                fid = info['fidelity_score']
                episode_fidelities.append(fid)
                if fid >= 0.8:  # Success threshold
                    episode_successes += 1

            if 'rescheduling_count' in info:
                episode_rescheduling = info['rescheduling_count']

        episode_rewards.append(episode_reward)
        if episode_fidelities:
            total_fidelities.append(np.mean(episode_fidelities))
        rescheduling_counts.append(episode_rescheduling)
        success_counts.append(episode_successes)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}: Reward={episode_reward:.6f}, "
                  f"Avg Fidelity={np.mean(episode_fidelities):.4f if episode_fidelities else 0:.4f}")

    # Calculate metrics
    results = {
        "policy": policy_name,
        "weights": {"w_time": w_time, "w_fidelity": w_fidelity},
        "num_episodes": num_episodes,
        "avg_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "avg_fidelity": float(np.mean(total_fidelities)) if total_fidelities else 0.0,
        "std_fidelity": float(np.std(total_fidelities)) if total_fidelities else 0.0,
        "avg_success_count": float(np.mean(success_counts)),
        "avg_rescheduling": float(np.mean(rescheduling_counts)),
        "episode_rewards": [float(r) for r in episode_rewards],
        "episode_fidelities": [float(f) for f in total_fidelities],
    }

    print(f"\nResults for {policy_name}:")
    print(f"  Avg Reward: {results['avg_reward']:.6f} ± {results['std_reward']:.6f}")
    print(f"  Avg Fidelity: {results['avg_fidelity']:.4f} ± {results['std_fidelity']:.4f}")
    print(f"  Avg Success Count: {results['avg_success_count']:.2f}")
    print(f"  Avg Rescheduling: {results['avg_rescheduling']:.2f}")

    return results


def load_optimized_weights(results_file: str = "./results/Hybrid_EA_DRL/optimization_results.json") -> tuple:
    """Load optimized weights from EA training results."""
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        weights = results["best_weights"]
        return weights["w_time"], weights["w_fidelity"]
    else:
        print(f"Warning: Results file not found at {results_file}")
        print("Using default weights: (0.7, 0.3)")
        return 0.7, 0.3


def plot_comparison(results_list: List[Dict], output_dir: str = "./results/Hybrid_EA_DRL"):
    """
    Create comparison plots for different policies.

    Args:
        results_list: List of evaluation results dictionaries
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Average Reward Comparison
    plt.figure(figsize=(10, 6))
    policies = [r["policy"] for r in results_list]
    rewards = [r["avg_reward"] for r in results_list]
    errors = [r["std_reward"] for r in results_list]

    plt.bar(policies, rewards, yerr=errors, capsize=5)
    plt.xlabel("Policy")
    plt.ylabel("Average Reward")
    plt.title("Policy Comparison: Average Reward")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_comparison.png"))
    plt.close()

    # Plot 2: Fidelity Comparison
    plt.figure(figsize=(10, 6))
    fidelities = [r["avg_fidelity"] for r in results_list]
    fid_errors = [r["std_fidelity"] for r in results_list]

    plt.bar(policies, fidelities, yerr=fid_errors, capsize=5, color='orange')
    plt.xlabel("Policy")
    plt.ylabel("Average Fidelity")
    plt.title("Policy Comparison: Average Fidelity")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fidelity_comparison.png"))
    plt.close()

    # Plot 3: Reward vs Fidelity Trade-off
    plt.figure(figsize=(10, 6))
    for r in results_list:
        plt.scatter(r["avg_reward"], r["avg_fidelity"], s=100, label=r["policy"])
        plt.annotate(r["policy"], (r["avg_reward"], r["avg_fidelity"]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.xlabel("Average Reward (Time Efficiency)")
    plt.ylabel("Average Fidelity (Success Probability)")
    plt.title("Reward vs Fidelity Trade-off")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_fidelity_tradeoff.png"))
    plt.close()

    print(f"\nPlots saved to {output_dir}/")


def main():
    """Main evaluation function."""
    print("\n" + "=" * 80)
    print("HYBRID EA+DRL EVALUATION")
    print("=" * 80 + "\n")

    # Load optimized weights (if available)
    opt_w_time, opt_w_fidelity = load_optimized_weights()

    # Evaluate different configurations
    results = []

    # 1. Baseline DRLQ (time-only, no fidelity consideration)
    results.append(evaluate_policy(
        "Baseline-DRLQ-TimeOnly",
        w_time=1.0,
        w_fidelity=0.0,
        num_episodes=20
    ))

    # 2. Hybrid EA+DRL with optimized weights
    results.append(evaluate_policy(
        "Hybrid-EA-DRL-Optimized",
        w_time=opt_w_time,
        w_fidelity=opt_w_fidelity,
        num_episodes=20
    ))

    # 3. Balanced weights (equal time and fidelity)
    results.append(evaluate_policy(
        "Balanced-50-50",
        w_time=0.5,
        w_fidelity=0.5,
        num_episodes=20
    ))

    # 4. Fidelity-focused
    results.append(evaluate_policy(
        "Fidelity-Focused",
        w_time=0.3,
        w_fidelity=0.7,
        num_episodes=20
    ))

    # Save results
    output_dir = "./results/Hybrid_EA_DRL"
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation results saved to: {results_file}")

    # Create comparison plots
    plot_comparison(results, output_dir)

    # Print summary table
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Policy':<30} {'Avg Reward':<15} {'Avg Fidelity':<15} {'Success Rate':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r['policy']:<30} {r['avg_reward']:<15.6f} {r['avg_fidelity']:<15.4f} "
              f"{r['avg_success_count']:<15.2f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

"""
Hybrid EA+DRL Training Script for Fidelity-Aware Quantum Task Scheduling

This script implements the complete hybrid system that combines:
- Evolutionary Algorithm (EA): Optimizes reward function weights
- Deep Reinforcement Learning (DRL): Learns optimal task placement policy

The system addresses the limitation of DRLQ by making it fidelity-aware,
balancing both time efficiency and quantum circuit success probability.
"""

import os
import ray
from ray import air, tune
from ray.rllib.algorithms.a2c import A2C, A2CConfig
from ray.tune.registry import register_env
import numpy as np
from typing import Dict, Tuple
import json
from datetime import datetime

from env_creator import env_creator
from env_wrapper import FidelityAwareRewardWrapper
from evolutionary_reward_shaping import EvolutionaryRewardShaper, EAConfig


# Configuration
DATASET_PATH = "./qdataset/qd_all.pickle"
RESULT_DIR = "./results/Hybrid_EA_DRL"
DRL_ALGORITHM = "A2C"  # Can be changed to "DQN"


def create_fidelity_aware_env(config):
    """
    Create environment with fidelity-aware reward shaping.

    Args:
        config: Environment configuration dict containing 'w_time' and 'w_fidelity'
    """
    # Get reward weights from config
    w_time = config.get("w_time", 0.7)
    w_fidelity = config.get("w_fidelity", 0.3)

    # Create base environment
    env_config = {
        "dataset": DATASET_PATH,
        "evaluation": config.get("evaluation", False),
        "policy": config.get("policy", "HybridEA-DRL"),
    }
    base_env = env_creator(env_config)

    # Wrap with fidelity-aware reward shaping
    env = FidelityAwareRewardWrapper(base_env, w_time=w_time, w_fidelity=w_fidelity)

    return env


def train_drl_agent(w_time: float, w_fidelity: float, num_episodes: int = 50) -> Tuple[float, Dict]:
    """
    Train a DRL agent with given reward weights.

    Args:
        w_time: Weight for time-based reward
        w_fidelity: Weight for fidelity-based reward
        num_episodes: Number of training episodes

    Returns:
        Tuple of (average_reward, info_dict)
    """
    print(f"\n{'='*60}")
    print(f"Training DRL Agent with w_time={w_time:.3f}, w_fidelity={w_fidelity:.3f}")
    print(f"{'='*60}\n")

    # Register environment
    env_name = f"QSimPy-Fidelity-{w_time:.3f}-{w_fidelity:.3f}"
    register_env(env_name, create_fidelity_aware_env)

    # Configure A2C algorithm
    config = (
        A2CConfig()
        .environment(
            env=env_name,
            env_config={
                "w_time": w_time,
                "w_fidelity": w_fidelity,
                "evaluation": False,
            }
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=2,
            rollout_fragment_length=200,
        )
        .training(
            gamma=0.9,
            lr=0.01,
            train_batch_size=400,
        )
        .evaluation(
            evaluation_interval=None,  # Disable built-in evaluation
        )
        .reporting(
            min_sample_timesteps_per_iteration=200,
        )
    )

    # Build agent
    algo = config.build()

    # Training loop
    episode_rewards = []
    episode_info_list = []

    try:
        for i in range(num_episodes):
            result = algo.train()

            # Extract metrics
            episode_reward = result.get("episode_reward_mean", 0)
            episode_rewards.append(episode_reward)

            # Store episode info
            info = {
                "iteration": i,
                "episode_reward_mean": episode_reward,
                "episodes_this_iter": result.get("episodes_this_iter", 0),
                "timesteps_total": result.get("timesteps_total", 0),
            }
            episode_info_list.append(info)

            if (i + 1) % 10 == 0:
                print(f"Episode {i+1}/{num_episodes}: Reward = {episode_reward:.6f}")

    finally:
        algo.stop()

    avg_reward = np.mean(episode_rewards) if episode_rewards else 0

    return avg_reward, {
        "episode_rewards": episode_rewards,
        "episode_info": episode_info_list,
        "avg_reward": avg_reward,
    }


def fitness_function_factory(ea_config: EAConfig):
    """
    Create fitness function for EA optimization.

    Returns:
        Function that evaluates fitness for given weights
    """
    def fitness_func(w_time: float, w_fidelity: float) -> float:
        """
        Evaluate fitness of weight configuration.

        Args:
            w_time: Weight for time reward
            w_fidelity: Weight for fidelity reward

        Returns:
            Fitness score (higher is better)
        """
        # Train DRL agent with these weights
        avg_reward, info = train_drl_agent(
            w_time, w_fidelity,
            num_episodes=ea_config.drl_episodes_per_eval
        )

        # Fitness is simply the average reward achieved
        # (could be extended with more sophisticated metrics)
        fitness = avg_reward

        return fitness

    return fitness_func


def main():
    """Main training loop for Hybrid EA+DRL system."""
    print("\n" + "=" * 80)
    print("HYBRID EA+DRL TRAINING FOR FIDELITY-AWARE QUANTUM TASK SCHEDULING")
    print("=" * 80)
    print(f"Dataset: {DATASET_PATH}")
    print(f"DRL Algorithm: {DRL_ALGORITHM}")
    print(f"Results Directory: {RESULT_DIR}")
    print("=" * 80 + "\n")

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Configure EA
    ea_config = EAConfig(
        population_size=6,  # Smaller population for faster iteration
        sigma0=0.2,
        max_generations=10,  # Reduced for initial testing
        drl_episodes_per_eval=30,  # Reduced for faster evaluation
        success_threshold=0.8,
    )

    # Initialize Evolutionary Reward Shaper
    ea = EvolutionaryRewardShaper(ea_config)

    # Create fitness function
    fitness_func = fitness_function_factory(ea_config)

    # Run EA optimization
    print("\nStarting Evolutionary Optimization...\n")
    start_time = datetime.now()

    best_weights = ea.optimize_weights(
        fitness_func=fitness_func,
        initial_weights=(0.7, 0.3),  # Start with time-focused weights
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Print results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Optimization Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Optimal Weights: w_time={best_weights[0]:.4f}, w_fidelity={best_weights[1]:.4f}")
    print(f"Best Fitness: {ea.best_fitness:.6f}")
    print(f"Total Generations: {ea.generation}")
    print("=" * 80 + "\n")

    # Save results
    os.makedirs(RESULT_DIR, exist_ok=True)

    results = {
        "best_weights": {
            "w_time": best_weights[0],
            "w_fidelity": best_weights[1],
        },
        "best_fitness": ea.best_fitness,
        "generations": ea.generation,
        "fitness_history": ea.fitness_history,
        "weight_history": [(w[0], w[1]) for w in ea.weight_history],
        "ea_config": {
            "population_size": ea_config.population_size,
            "sigma0": ea_config.sigma0,
            "max_generations": ea_config.max_generations,
            "drl_episodes_per_eval": ea_config.drl_episodes_per_eval,
        },
        "duration_seconds": duration,
        "timestamp": start_time.isoformat(),
    }

    results_file = os.path.join(RESULT_DIR, "optimization_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_file}")

    # Train final agent with optimized weights
    print("\n" + "=" * 80)
    print("TRAINING FINAL AGENT WITH OPTIMIZED WEIGHTS")
    print("=" * 80 + "\n")

    final_avg_reward, final_info = train_drl_agent(
        best_weights[0],
        best_weights[1],
        num_episodes=100  # More episodes for final training
    )

    print(f"\nFinal Agent Performance: {final_avg_reward:.6f}")

    # Save final agent info
    final_results = {
        "weights": {"w_time": best_weights[0], "w_fidelity": best_weights[1]},
        "final_avg_reward": final_avg_reward,
        "episode_rewards": final_info["episode_rewards"],
    }

    final_results_file = os.path.join(RESULT_DIR, "final_agent_results.json")
    with open(final_results_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"Final agent results saved to: {final_results_file}\n")

    # Shutdown Ray
    ray.shutdown()

    return best_weights, ea.best_fitness


if __name__ == "__main__":
    best_weights, best_fitness = main()
    print("\n✓ Hybrid EA+DRL training completed successfully!")
    print(f"✓ Best weights: w_time={best_weights[0]:.4f}, w_fidelity={best_weights[1]:.4f}")
    print(f"✓ Best fitness: {best_fitness:.6f}\n")

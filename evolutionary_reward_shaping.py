"""
Evolutionary Reward Shaping for Quantum Task Scheduling
Uses CMA-ES to optimize reward function weights for fidelity-aware scheduling.

This module implements the Evolutionary Algorithm (EA) component of the Hybrid EA+DRL system.
The EA tunes the reward weights (w_time, w_fidelity) by evaluating different configurations
through DRL agent performance.
"""

import numpy as np
from typing import Tuple, Callable, Dict, List
import cma
from dataclasses import dataclass


@dataclass
class EAConfig:
    """Configuration for the Evolutionary Algorithm."""
    population_size: int = 8  # Number of weight configurations to evaluate per generation
    sigma0: float = 0.2  # Initial standard deviation for CMA-ES
    max_generations: int = 20  # Maximum number of EA generations
    drl_episodes_per_eval: int = 50  # Number of DRL episodes to evaluate each weight config
    success_threshold: float = 0.8  # Minimum fidelity threshold for "successful" tasks


class EvolutionaryRewardShaper:
    """
    Evolutionary Algorithm for optimizing reward function weights.

    Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to find optimal
    balance between time efficiency and fidelity in the reward function.

    The EA evolves weight configurations [w_time, w_fidelity] by:
    1. Generating population of weight candidates
    2. Training DRL agent with each weight configuration
    3. Evaluating fitness based on global metrics (throughput + success rate)
    4. Selecting best configurations for next generation
    """

    def __init__(self, config: EAConfig = None):
        """
        Initialize the Evolutionary Reward Shaper.

        Args:
            config: Configuration parameters for the EA
        """
        self.config = config if config else EAConfig()
        self.generation = 0
        self.best_weights = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        self.weight_history = []

    def optimize_weights(
        self,
        fitness_func: Callable[[float, float], float],
        initial_weights: Tuple[float, float] = (0.7, 0.3),
    ) -> Tuple[float, float]:
        """
        Optimize reward weights using CMA-ES.

        Args:
            fitness_func: Function that takes (w_time, w_fidelity) and returns fitness score
            initial_weights: Starting point for optimization

        Returns:
            Tuple of optimized (w_time, w_fidelity) weights
        """
        # Initialize CMA-ES optimizer
        # We optimize weights that must sum to 1, so we optimize w_fidelity in [0, 1]
        # and set w_time = 1 - w_fidelity
        opts = {
            'popsize': self.config.population_size,
            'maxiter': self.config.max_generations,
            'bounds': [0, 1],  # w_fidelity must be in [0, 1]
            'verbose': -1,  # Suppress CMA-ES output
        }

        # Initial guess: just optimize w_fidelity
        x0 = initial_weights[1]  # Start with initial w_fidelity

        # Create optimizer
        es = cma.CMAEvolutionStrategy(x0, self.config.sigma0, opts)

        print("\n" + "=" * 80)
        print("EVOLUTIONARY REWARD SHAPING - CMA-ES Optimization")
        print("=" * 80)
        print(f"Population Size: {self.config.population_size}")
        print(f"Max Generations: {self.config.max_generations}")
        print(f"DRL Episodes per Evaluation: {self.config.drl_episodes_per_eval}")
        print(f"Initial Weights: w_time={initial_weights[0]:.3f}, w_fidelity={initial_weights[1]:.3f}")
        print("=" * 80 + "\n")

        # Evolution loop
        while not es.stop():
            self.generation += 1

            # Get candidate solutions (w_fidelity values)
            solutions = es.ask()

            # Evaluate fitness for each solution
            fitnesses = []
            for i, w_fid in enumerate(solutions):
                # Ensure w_fidelity is in valid range
                w_fid = np.clip(w_fid, 0, 1)
                w_time = 1.0 - w_fid

                print(f"Generation {self.generation}, Individual {i+1}/{len(solutions)}: "
                      f"w_time={w_time:.3f}, w_fidelity={w_fid:.3f}")

                # Evaluate this weight configuration
                fitness = fitness_func(w_time, w_fid)
                fitnesses.append(-fitness)  # CMA-ES minimizes, so negate for maximization

                print(f"  → Fitness: {fitness:.6f}")

                # Track best configuration
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_weights = (w_time, w_fid)
                    print(f"  ✓ New best fitness: {fitness:.6f}")

            # Update CMA-ES with fitness results
            es.tell(solutions, fitnesses)

            # Log generation summary
            avg_fitness = -np.mean(fitnesses)
            print(f"\nGeneration {self.generation} Summary:")
            print(f"  Best Fitness: {self.best_fitness:.6f}")
            print(f"  Avg Fitness: {avg_fitness:.6f}")
            print(f"  Best Weights: w_time={self.best_weights[0]:.3f}, "
                  f"w_fidelity={self.best_weights[1]:.3f}\n")

            self.fitness_history.append(self.best_fitness)
            self.weight_history.append(self.best_weights)

        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"Best Weights: w_time={self.best_weights[0]:.3f}, w_fidelity={self.best_weights[1]:.3f}")
        print(f"Best Fitness: {self.best_fitness:.6f}")
        print("=" * 80 + "\n")

        return self.best_weights

    def calculate_fitness(
        self,
        episode_rewards: List[float],
        episode_info: List[Dict],
    ) -> float:
        """
        Calculate fitness score for a weight configuration.

        Fitness combines:
        - Task completion throughput (higher is better)
        - Success rate based on fidelity (higher is better)
        - Constraint satisfaction (minimize rescheduling)

        Args:
            episode_rewards: List of total rewards from each episode
            episode_info: List of info dicts from each episode

        Returns:
            Fitness score (higher is better)
        """
        # Average episode reward (reflects time efficiency)
        avg_reward = np.mean(episode_rewards)

        # Calculate success rate (tasks with fidelity > threshold)
        total_tasks = 0
        successful_tasks = 0
        total_fidelity = 0
        total_rescheduling = 0

        for info in episode_info:
            if 'fidelity_score' in info:
                total_tasks += 1
                fid = info['fidelity_score']
                total_fidelity += fid

                if fid >= self.config.success_threshold:
                    successful_tasks += 1

            if 'rescheduling_count' in info:
                total_rescheduling += info['rescheduling_count']

        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
        avg_fidelity = total_fidelity / total_tasks if total_tasks > 0 else 0
        avg_rescheduling = total_rescheduling / len(episode_info) if episode_info else 0

        # Combined fitness: balance throughput and quality
        # Higher reward = faster completion
        # Higher success rate = better quality outputs
        # Lower rescheduling = better constraint satisfaction
        fitness = (
            0.4 * avg_reward +  # Time efficiency
            0.4 * success_rate +  # Quality (success rate)
            0.2 * avg_fidelity -  # Quality (avg fidelity)
            0.1 * avg_rescheduling  # Stability (minimize rescheduling)
        )

        return fitness

    def get_evolution_summary(self) -> Dict:
        """
        Get summary of evolution process.

        Returns:
            Dictionary containing evolution history and best results
        """
        return {
            'generations': self.generation,
            'best_weights': self.best_weights,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'weight_history': self.weight_history,
        }

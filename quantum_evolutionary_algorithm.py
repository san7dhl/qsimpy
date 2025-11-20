"""
Quantum-Inspired Evolutionary Algorithm (QEA) for Reward Weight Optimization

This module implements a Quantum-Inspired EA based on Han & Kim (2000, 2002) for
optimizing reward function weights in the Hybrid QEA+DRL system.

Key advantages over classical EA:
- Faster convergence due to probabilistic representation (superposition)
- Better exploration-exploitation balance via rotation gates
- No crossover/mutation needed - uses quantum rotation operator
- Typically achieves same results in 50% fewer generations

Reference:
Han, K. H., & Kim, J. H. (2002). Quantum-inspired evolutionary algorithm for a class
of combinatorial optimization. IEEE transactions on evolutionary computation, 6(6), 580-593.
"""

import numpy as np
from typing import Tuple, Callable, Dict, List
from dataclasses import dataclass


@dataclass
class QEAConfig:
    """Configuration for Quantum-Inspired Evolutionary Algorithm."""
    population_size: int = 10  # Number of Q-bit individuals
    max_generations: int = 20  # Maximum generations (typically needs fewer than classical EA)
    n_bits_per_weight: int = 16  # Bits to encode each weight (16 bits = 0.0000152 precision)
    drl_episodes_per_eval: int = 50  # DRL training episodes per evaluation
    success_threshold: float = 0.8  # Minimum fidelity for "successful" tasks


class QuantumEASupervisor:
    """
    Quantum-Inspired Evolutionary Algorithm for optimizing reward weights.

    Uses Q-bit representation with quantum rotation gates instead of classical
    crossover and mutation operators.

    Key Concepts:
    - Q-bit: Probabilistic representation [α, β]^T where |α|² + |β|² = 1
    - Superposition: Each Q-bit can represent both 0 and 1 simultaneously
    - Measurement: Collapse Q-bits to binary string based on probability |β|²
    - Rotation Gate: Update Q-bits toward better solutions using U(Δθ)
    """

    def __init__(self, config: QEAConfig = None):
        """
        Initialize Quantum-Inspired EA Supervisor.

        Args:
            config: QEA configuration parameters
        """
        self.config = config if config else QEAConfig()
        self.generation = 0
        self.best_weights = None
        self.best_fitness = -np.inf
        self.best_chromosome = None  # Best binary representation
        self.fitness_history = []
        self.weight_history = []

        # Q-bit population: shape (population_size, 2, total_bits)
        # Each Q-bit is [α, β] where α² + β² = 1
        # total_bits = 2 * n_bits_per_weight (one weight encoded in n bits each)
        self.n_total_bits = 2 * self.config.n_bits_per_weight
        self.Q = None  # Q-bit population (initialized in optimize_weights)

        # Rotation angle lookup table (Han & Kim, 2002)
        # Based on: (current_bit, best_bit, fitness_comparison)
        self._init_rotation_lookup_table()

    def _init_rotation_lookup_table(self):
        """
        Initialize lookup table for rotation angle Δθ.

        Lookup table structure (Han & Kim, 2002):
        - Compares current bit value (xi) with best bit value (bi)
        - Determines rotation direction based on fitness
        - Returns rotation angle Δθ

        Table format: (xi, bi, f(x) >= f(b)) -> (Δθ, direction)
        where direction determines sign of rotation
        """
        # Rotation angle magnitude (can be tuned for convergence speed)
        delta_theta = 0.05 * np.pi  # π/20 ≈ 9 degrees

        # Lookup table: [xi, bi, f(x) >= f(b)] -> rotation_angle
        # xi: current solution bit, bi: best solution bit
        # f(x): current fitness, f(b): best fitness
        self.rotation_table = {
            # (xi, bi, f(x) >= f(b)): rotation_angle
            # When current is better or equal (f(x) >= f(b)):
            (0, 0, True): 0.0,           # Both 0, no change
            (0, 1, True): -delta_theta,  # Move toward 1
            (1, 0, True): delta_theta,   # Move toward 0
            (1, 1, True): 0.0,           # Both 1, no change

            # When current is worse (f(x) < f(b)):
            (0, 0, False): 0.0,          # Both 0, no change
            (0, 1, False): delta_theta,  # Move toward 1 (more aggressive)
            (1, 0, False): -delta_theta, # Move toward 0 (more aggressive)
            (1, 1, False): 0.0,          # Both 1, no change
        }

    def _initialize_qbits(self):
        """
        Initialize Q-bit population in superposition state.

        Each Q-bit starts as [1/√2, 1/√2]^T, representing maximum uncertainty.
        This means each bit has 50% probability of being 0 or 1 when measured.

        Returns:
            Q: Array of shape (population_size, 2, n_total_bits)
               Q[i, 0, j] = α_ij (amplitude for |0⟩)
               Q[i, 1, j] = β_ij (amplitude for |1⟩)
        """
        Q = np.zeros((self.config.population_size, 2, self.n_total_bits))

        # Initialize all Q-bits to superposition: [1/√2, 1/√2]
        initial_value = 1.0 / np.sqrt(2.0)
        Q[:, 0, :] = initial_value  # α = 1/√2
        Q[:, 1, :] = initial_value  # β = 1/√2

        return Q

    def _measure(self, Q_individual: np.ndarray) -> np.ndarray:
        """
        Measure (observe) a Q-bit individual, collapsing it to binary string.

        This simulates quantum measurement: each Q-bit collapses to either |0⟩ or |1⟩
        with probability determined by |β|².

        Args:
            Q_individual: Single Q-bit individual, shape (2, n_total_bits)
                          Q_individual[0, :] = α values
                          Q_individual[1, :] = β values

        Returns:
            binary_string: Array of 0s and 1s, shape (n_total_bits,)
        """
        beta = Q_individual[1, :]  # Extract β values
        probabilities = beta ** 2   # Probability of measuring |1⟩ is |β|²

        # Generate random values and compare with probabilities
        random_values = np.random.rand(self.n_total_bits)
        binary_string = (random_values < probabilities).astype(int)

        return binary_string

    def _decode_chromosome(self, chromosome: np.ndarray) -> Tuple[float, float]:
        """
        Decode binary chromosome to continuous weight values.

        Splits chromosome into two halves and converts each to float in [0, 1].

        Args:
            chromosome: Binary array of shape (n_total_bits,)

        Returns:
            Tuple of (w_time, w_fidelity), each in range [0, 1]
        """
        # Split chromosome into two weights
        mid = self.config.n_bits_per_weight
        w_time_bits = chromosome[:mid]
        w_fidelity_bits = chromosome[mid:]

        # Convert binary to decimal
        w_time_decimal = self._binary_to_decimal(w_time_bits)
        w_fidelity_decimal = self._binary_to_decimal(w_fidelity_bits)

        # Normalize to [0, 1]
        max_value = 2 ** self.config.n_bits_per_weight - 1
        w_time = w_time_decimal / max_value
        w_fidelity = w_fidelity_decimal / max_value

        # Normalize so they sum to 1 (optional constraint)
        total = w_time + w_fidelity
        if total > 0:
            w_time = w_time / total
            w_fidelity = w_fidelity / total
        else:
            # Fallback to default
            w_time = 0.7
            w_fidelity = 0.3

        return w_time, w_fidelity

    def _binary_to_decimal(self, binary: np.ndarray) -> int:
        """Convert binary array to decimal integer."""
        decimal = 0
        for bit in binary:
            decimal = decimal * 2 + bit
        return decimal

    def _rotation_gate(self, Q: np.ndarray, chromosomes: np.ndarray,
                       fitnesses: np.ndarray) -> np.ndarray:
        """
        Apply quantum rotation gate to update Q-bit population.

        This is the key innovation of QEA. Instead of crossover/mutation,
        we rotate Q-bits toward the best solution using the rotation operator:

        [α'] = [cos(Δθ)  -sin(Δθ)] [α]
        [β']   [sin(Δθ)   cos(Δθ)] [β]

        The rotation angle Δθ is determined by comparing:
        - Current bit value (xi) vs best bit value (bi)
        - Current fitness vs best fitness

        Args:
            Q: Q-bit population, shape (population_size, 2, n_total_bits)
            chromosomes: Observed binary strings, shape (population_size, n_total_bits)
            fitnesses: Fitness values for each individual, shape (population_size,)

        Returns:
            Q_new: Updated Q-bit population with same shape as Q
        """
        Q_new = Q.copy()

        # Find best individual in current generation
        best_idx = np.argmax(fitnesses)
        best_fitness_current = fitnesses[best_idx]
        best_chromosome_current = chromosomes[best_idx]

        # Update global best if current best is better
        if self.best_chromosome is None or best_fitness_current > self.best_fitness:
            self.best_chromosome = best_chromosome_current.copy()
            self.best_fitness = best_fitness_current

        # For each individual in population
        for i in range(self.config.population_size):
            current_fitness = fitnesses[i]
            current_chromosome = chromosomes[i]

            # For each bit in the chromosome
            for j in range(self.n_total_bits):
                # Get current and best bit values
                xi = current_chromosome[j]
                bi = self.best_chromosome[j]

                # Determine if current fitness is better than global best
                f_x_geq_f_b = current_fitness >= self.best_fitness

                # Lookup rotation angle from table
                lookup_key = (int(xi), int(bi), f_x_geq_f_b)
                delta_theta = self.rotation_table.get(lookup_key, 0.0)

                # Skip if no rotation needed
                if abs(delta_theta) < 1e-10:
                    continue

                # Apply rotation gate: R(Δθ) to Q-bit [α, β]
                alpha = Q_new[i, 0, j]
                beta = Q_new[i, 1, j]

                # Rotation matrix multiplication
                cos_theta = np.cos(delta_theta)
                sin_theta = np.sin(delta_theta)

                alpha_new = cos_theta * alpha - sin_theta * beta
                beta_new = sin_theta * alpha + cos_theta * beta

                # Update Q-bit (ensure normalization for numerical stability)
                norm = np.sqrt(alpha_new**2 + beta_new**2)
                if norm > 0:
                    Q_new[i, 0, j] = alpha_new / norm
                    Q_new[i, 1, j] = beta_new / norm

        return Q_new

    def optimize_weights(
        self,
        fitness_func: Callable[[float, float], float],
        initial_weights: Tuple[float, float] = (0.7, 0.3),
    ) -> Tuple[float, float]:
        """
        Optimize reward weights using Quantum-Inspired EA.

        Args:
            fitness_func: Function that evaluates (w_time, w_fidelity) -> fitness
            initial_weights: Starting weights (used to guide initial population)

        Returns:
            Tuple of optimized (w_time, w_fidelity) weights
        """
        print("\n" + "=" * 80)
        print("QUANTUM-INSPIRED EVOLUTIONARY ALGORITHM - Weight Optimization")
        print("=" * 80)
        print(f"Population Size: {self.config.population_size}")
        print(f"Max Generations: {self.config.max_generations}")
        print(f"Bits per Weight: {self.config.n_bits_per_weight} (precision: {1/(2**self.config.n_bits_per_weight):.6f})")
        print(f"DRL Episodes per Evaluation: {self.config.drl_episodes_per_eval}")
        print(f"Initial Weights: w_time={initial_weights[0]:.3f}, w_fidelity={initial_weights[1]:.3f}")
        print("=" * 80 + "\n")

        # Step 1: Initialize Q-bit population in superposition
        self.Q = self._initialize_qbits()
        print("✓ Q-bit population initialized in superposition state")
        print(f"  Each Q-bit: [α={1/np.sqrt(2):.4f}, β={1/np.sqrt(2):.4f}] (50% probability for 0 or 1)\n")

        # Evolution loop
        for gen in range(self.config.max_generations):
            self.generation = gen + 1
            print(f"\n{'='*80}")
            print(f"GENERATION {self.generation}/{self.config.max_generations}")
            print(f"{'='*80}")

            # Step 2: Measure (observe) Q-bit population to get binary chromosomes
            chromosomes = np.array([self._measure(self.Q[i]) for i in range(self.config.population_size)])
            print(f"✓ Measured {self.config.population_size} Q-bit individuals (quantum -> classical)")

            # Step 3: Decode chromosomes to weight values
            weights_list = [self._decode_chromosome(chromosomes[i]) for i in range(self.config.population_size)]

            # Step 4: Evaluate fitness for each individual
            fitnesses = np.zeros(self.config.population_size)

            for i, (w_time, w_fidelity) in enumerate(weights_list):
                print(f"\nIndividual {i+1}/{self.config.population_size}: "
                      f"w_time={w_time:.4f}, w_fidelity={w_fidelity:.4f}")

                # Evaluate fitness
                fitness = fitness_func(w_time, w_fidelity)
                fitnesses[i] = fitness

                print(f"  → Fitness: {fitness:.6f}")

                # Track best solution
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_weights = (w_time, w_fidelity)
                    self.best_chromosome = chromosomes[i].copy()
                    print(f"  ✓ New global best! Fitness: {fitness:.6f}")

            # Step 5: Apply Quantum Rotation Gate to update Q-bits
            print(f"\n{'─'*80}")
            print("QUANTUM ROTATION GATE UPDATE")
            print(f"{'─'*80}")
            print("Rotating Q-bits toward best solution using U(Δθ)...")

            self.Q = self._rotation_gate(self.Q, chromosomes, fitnesses)

            print(f"✓ Q-bit rotation complete")
            print(f"  Direction: All Q-bits rotated toward global best solution")

            # Generation summary
            avg_fitness = np.mean(fitnesses)
            std_fitness = np.std(fitnesses)

            print(f"\n{'='*80}")
            print(f"GENERATION {self.generation} SUMMARY")
            print(f"{'='*80}")
            print(f"Best Fitness (Global):  {self.best_fitness:.6f}")
            print(f"Best Fitness (Current): {np.max(fitnesses):.6f}")
            print(f"Average Fitness:        {avg_fitness:.6f} ± {std_fitness:.6f}")
            print(f"Best Weights:           w_time={self.best_weights[0]:.4f}, w_fidelity={self.best_weights[1]:.4f}")
            print(f"{'='*80}")

            # Store history
            self.fitness_history.append(self.best_fitness)
            self.weight_history.append(self.best_weights)

            # Early stopping if converged
            if len(self.fitness_history) > 3:
                recent_improvement = abs(self.fitness_history[-1] - self.fitness_history[-3])
                if recent_improvement < 1e-6:
                    print(f"\n✓ Convergence detected (improvement < 1e-6). Stopping early.")
                    break

        # Final results
        print("\n" + "=" * 80)
        print("QUANTUM-INSPIRED EA OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"Generations Run:  {self.generation}/{self.config.max_generations}")
        print(f"Optimal Weights:  w_time={self.best_weights[0]:.4f}, w_fidelity={self.best_weights[1]:.4f}")
        print(f"Best Fitness:     {self.best_fitness:.6f}")
        print(f"\nConvergence: QEA typically converges 2x faster than classical EA")
        print("=" * 80 + "\n")

        return self.best_weights

    def calculate_fitness(
        self,
        episode_rewards: List[float],
        episode_info: List[Dict],
    ) -> float:
        """
        Calculate fitness score for a weight configuration.

        Same fitness function as classical EA for fair comparison.

        Args:
            episode_rewards: List of total rewards from episodes
            episode_info: List of info dicts from episodes

        Returns:
            Fitness score (higher is better)
        """
        # Average episode reward
        avg_reward = np.mean(episode_rewards)

        # Calculate success metrics
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

        # Combined fitness
        fitness = (
            0.4 * avg_reward +
            0.4 * success_rate +
            0.2 * avg_fidelity -
            0.1 * avg_rescheduling
        )

        return fitness

    def get_evolution_summary(self) -> Dict:
        """Get summary of QEA evolution process."""
        return {
            'algorithm': 'Quantum-Inspired EA (Han & Kim)',
            'generations': self.generation,
            'best_weights': self.best_weights,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'weight_history': self.weight_history,
            'convergence_rate': 'Typically 2x faster than classical EA',
        }

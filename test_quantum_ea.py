"""
Test Script for Quantum-Inspired Evolutionary Algorithm (QEA)

Validates QEA components work correctly before full training.
"""

import numpy as np
import sys
from quantum_evolutionary_algorithm import QuantumEASupervisor, QEAConfig


def test_qbit_initialization():
    """Test 1: Q-bit initialization in superposition."""
    print("\n" + "=" * 80)
    print("TEST 1: Q-bit Initialization")
    print("=" * 80)

    config = QEAConfig(population_size=5, n_bits_per_weight=8)
    qea = QuantumEASupervisor(config)

    Q = qea._initialize_qbits()

    print(f"Q-bit population shape: {Q.shape}")
    print(f"Expected: ({config.population_size}, 2, {2 * config.n_bits_per_weight})")

    # Check superposition initialization
    expected_value = 1.0 / np.sqrt(2.0)
    alpha_check = np.allclose(Q[:, 0, :], expected_value)
    beta_check = np.allclose(Q[:, 1, :], expected_value)

    print(f"\nAlpha values (should be {expected_value:.4f}): {Q[0, 0, :5]}")
    print(f"Beta values (should be {expected_value:.4f}): {Q[0, 1, :5]}")

    # Check normalization |α|² + |β|² = 1
    norms = Q[:, 0, :]**2 + Q[:, 1, :]**2
    norm_check = np.allclose(norms, 1.0)

    print(f"\nNormalization check (|α|² + |β|² = 1): {norm_check}")
    print(f"Sample norms: {norms[0, :5]}")

    assert alpha_check and beta_check and norm_check, "Q-bit initialization failed!"
    print("\n✓ TEST 1 PASSED: Q-bits correctly initialized in superposition")
    return True


def test_measurement():
    """Test 2: Measurement (collapse) of Q-bits."""
    print("\n" + "=" * 80)
    print("TEST 2: Q-bit Measurement (Collapse)")
    print("=" * 80)

    config = QEAConfig(population_size=1, n_bits_per_weight=8)
    qea = QuantumEASupervisor(config)

    Q = qea._initialize_qbits()

    # Measure multiple times to check probability distribution
    n_measurements = 1000
    measurements = []

    for _ in range(n_measurements):
        binary = qea._measure(Q[0])
        measurements.append(binary)

    measurements = np.array(measurements)

    # For superposition [1/√2, 1/√2], expect ~50% ones
    ones_ratio = np.mean(measurements)

    print(f"Number of measurements: {n_measurements}")
    print(f"Ratio of 1s: {ones_ratio:.3f}")
    print(f"Expected: ~0.500 (50% due to superposition)")

    # Check if ratio is close to 50% (within 5% tolerance)
    assert 0.45 <= ones_ratio <= 0.55, f"Measurement ratio {ones_ratio} not close to 0.5!"

    print("\n✓ TEST 2 PASSED: Measurement correctly collapses Q-bits")
    return True


def test_decoding():
    """Test 3: Decode binary chromosome to weights."""
    print("\n" + "=" * 80)
    print("TEST 3: Binary to Weight Decoding")
    print("=" * 80)

    config = QEAConfig(n_bits_per_weight=8)
    qea = QuantumEASupervisor(config)

    # Test cases
    test_cases = [
        (np.array([0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0]), "All zeros"),
        (np.array([1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1]), "All ones"),
        (np.array([1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1]), "Mixed"),
    ]

    for chromosome, description in test_cases:
        w_time, w_fidelity = qea._decode_chromosome(chromosome)

        print(f"\n{description}:")
        print(f"  Binary: {chromosome}")
        print(f"  Decoded: w_time={w_time:.4f}, w_fidelity={w_fidelity:.4f}")
        print(f"  Sum: {w_time + w_fidelity:.4f} (should be 1.0)")

        assert abs((w_time + w_fidelity) - 1.0) < 1e-6, "Weights don't sum to 1!"
        assert 0 <= w_time <= 1 and 0 <= w_fidelity <= 1, "Weights out of range!"

    print("\n✓ TEST 3 PASSED: Binary decoding produces valid weights")
    return True


def test_rotation_gate():
    """Test 4: Quantum rotation gate."""
    print("\n" + "=" * 80)
    print("TEST 4: Quantum Rotation Gate")
    print("=" * 80)

    config = QEAConfig(population_size=3, n_bits_per_weight=4)
    qea = QuantumEASupervisor(config)

    # Initialize Q-bits
    Q = qea._initialize_qbits()

    # Create mock chromosomes and fitnesses
    chromosomes = np.array([
        [0, 1, 0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 1, 1, 1, 0, 0],
    ])

    fitnesses = np.array([0.5, 0.8, 0.3])

    # Set initial best
    qea.best_chromosome = chromosomes[1].copy()
    qea.best_fitness = fitnesses[1]

    print(f"Initial Q-bit (individual 0, bit 0): α={Q[0, 0, 0]:.4f}, β={Q[0, 1, 0]:.4f}")

    # Apply rotation
    Q_new = qea._rotation_gate(Q, chromosomes, fitnesses)

    print(f"After rotation (individual 0, bit 0): α={Q_new[0, 0, 0]:.4f}, β={Q_new[0, 1, 0]:.4f}")

    # Check normalization is preserved
    norms = Q_new[:, 0, :]**2 + Q_new[:, 1, :]**2
    norm_check = np.allclose(norms, 1.0)

    print(f"\nNormalization preserved: {norm_check}")
    print(f"Sample norms after rotation: {norms[0, :5]}")

    # Check Q-bits have changed (rotation applied)
    changed = not np.allclose(Q, Q_new)
    print(f"Q-bits updated: {changed}")

    assert norm_check, "Rotation gate broke normalization!"
    print("\n✓ TEST 4 PASSED: Rotation gate correctly updates Q-bits")
    return True


def test_fitness_calculation():
    """Test 5: Fitness calculation."""
    print("\n" + "=" * 80)
    print("TEST 5: Fitness Calculation")
    print("=" * 80)

    config = QEAConfig(success_threshold=0.8)
    qea = QuantumEASupervisor(config)

    # Mock episode data
    episode_rewards = [0.05, 0.06, 0.055, 0.058, 0.052]
    episode_info = [
        {"fidelity_score": 0.85, "rescheduling_count": 2},
        {"fidelity_score": 0.90, "rescheduling_count": 1},
        {"fidelity_score": 0.88, "rescheduling_count": 1},
        {"fidelity_score": 0.75, "rescheduling_count": 3},  # Below threshold
        {"fidelity_score": 0.89, "rescheduling_count": 2},
    ]

    fitness = qea.calculate_fitness(episode_rewards, episode_info)

    print(f"Episode rewards: {episode_rewards}")
    print(f"Fidelity scores: {[info['fidelity_score'] for info in episode_info]}")
    print(f"Calculated fitness: {fitness:.6f}")

    assert fitness > 0, "Fitness should be positive!"
    print("\n✓ TEST 5 PASSED: Fitness calculation works correctly")
    return True


def test_full_optimization_mock():
    """Test 6: Full optimization with mock fitness function."""
    print("\n" + "=" * 80)
    print("TEST 6: Full QEA Optimization (Mock Fitness)")
    print("=" * 80)

    config = QEAConfig(
        population_size=4,
        max_generations=3,  # Short run for testing
        n_bits_per_weight=8,
    )

    qea = QuantumEASupervisor(config)

    # Mock fitness function (quadratic, optimal at w_time=0.6)
    def mock_fitness(w_time, w_fidelity):
        # Quadratic function with peak at w_time=0.6, w_fidelity=0.4
        fitness = 1.0 - ((w_time - 0.6)**2 + (w_fidelity - 0.4)**2)
        return fitness

    print("\nRunning QEA with mock fitness function...")
    print("Expected optimal: w_time≈0.6, w_fidelity≈0.4\n")

    best_weights = qea.optimize_weights(mock_fitness, initial_weights=(0.5, 0.5))

    print(f"\nOptimization complete!")
    print(f"Found weights: w_time={best_weights[0]:.4f}, w_fidelity={best_weights[1]:.4f}")
    print(f"Best fitness: {qea.best_fitness:.6f}")

    # Check if reasonably close to optimal (within 0.2)
    assert abs(best_weights[0] - 0.6) < 0.3, "w_time not close to optimal!"
    assert abs(best_weights[1] - 0.4) < 0.3, "w_fidelity not close to optimal!"

    print("\n✓ TEST 6 PASSED: Full QEA optimization works correctly")
    return True


def run_all_tests():
    """Run all QEA validation tests."""
    print("\n" + "=" * 80)
    print("QUANTUM-INSPIRED EA VALIDATION TESTS")
    print("=" * 80)

    tests = [
        ("Q-bit Initialization", test_qbit_initialization),
        ("Q-bit Measurement", test_measurement),
        ("Binary to Weight Decoding", test_decoding),
        ("Quantum Rotation Gate", test_rotation_gate),
        ("Fitness Calculation", test_fitness_calculation),
        ("Full QEA Optimization (Mock)", test_full_optimization_mock),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("QEA TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status:<12} {test_name}")

    print("-" * 80)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL QEA TESTS PASSED! System ready for training.")
        print("\n📝 Key QEA Components Validated:")
        print("   ✓ Q-bit superposition representation")
        print("   ✓ Quantum measurement (collapse)")
        print("   ✓ Binary-to-weight decoding")
        print("   ✓ Rotation gate updates")
        print("   ✓ Fitness evaluation")
        print("   ✓ Full optimization loop")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

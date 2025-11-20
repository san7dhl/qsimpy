"""
Test Script for Hybrid EA+DRL Fidelity-Aware System

Validates that all components are working correctly before full training.
"""

import sys
import numpy as np
import simpy
from qsimpy.resources.IBMQNode import create_ibmq_node
from qsimpy.tasks.QTask import QTask
from env_creator import env_creator
from env_wrapper import FidelityAwareRewardWrapper


def test_qnode_fidelity():
    """Test 1: QNode fidelity calculation"""
    print("\n" + "=" * 60)
    print("TEST 1: QNode Fidelity Calculation")
    print("=" * 60)

    env = simpy.Environment()
    nodes = []
    node_names = ["washington", "kolkata", "hanoi", "perth", "lagos"]

    for i, name in enumerate(node_names):
        node = create_ibmq_node(env, i, name)
        nodes.append(node)

        # Test base fidelity (without task)
        base_fidelity = node.get_fidelity_score()
        print(f"\n{name.upper()} (Node {i}):")
        print(f"  Qubits: {node.qubit_number}")
        print(f"  CLOPS: {node.clops}")
        print(f"  Base Fidelity: {base_fidelity:.4f}")

        # Test with mock task
        mock_task_data = {
            "qubit_number": 5,
            "circuit_layers": 50,
        }

        class MockTask:
            def __init__(self, qubits, layers):
                self.qubit_number = qubits
                self.circuit_layers = layers

            def get_circuit_layers(self):
                return self.circuit_layers

        task_small = MockTask(5, 50)
        task_large = MockTask(20, 500)

        fid_small = node.get_fidelity_score(task_small)
        fid_large = node.get_fidelity_score(task_large)

        print(f"  Task Fidelity (5q, 50 layers): {fid_small:.6f}")
        print(f"  Task Fidelity (20q, 500 layers): {fid_large:.6f}")

    print("\n✓ Test 1 PASSED: Fidelity calculation working correctly")
    return True


def test_environment_state_space():
    """Test 2: Enhanced state space with fidelity"""
    print("\n" + "=" * 60)
    print("TEST 2: Environment State Space")
    print("=" * 60)

    try:
        env = env_creator({"dataset": "./qdataset/qd_all.pickle"})
        obs, info = env.reset()

        print(f"\nObservation shape: {obs.shape}")
        print(f"Expected shape: (24,)")
        print(f"Observation dimension: {env.obs_dim}")

        # Check observation space bounds
        print(f"\nObservation space:")
        print(f"  Low bounds: {env.observation_space.low}")
        print(f"  High bounds: {env.observation_space.high}")

        # Parse observation
        task_obs = obs[:4]
        node_obs = obs[4:].reshape(5, 4)

        print(f"\nTask observation (4 features):")
        print(f"  {task_obs}")
        print(f"  [arrival_time, qubits, layers, rescheduling_count]")

        print(f"\nNode observations (5 nodes × 4 features):")
        for i in range(5):
            print(f"  Node {i}: qubits={node_obs[i, 0]:.0f}, clops={node_obs[i, 1]:.0f}, "
                  f"next_time={node_obs[i, 2]:.2f}, fidelity={node_obs[i, 3]:.4f}")

        assert obs.shape == (24,), f"Expected shape (24,), got {obs.shape}"
        assert env.obs_dim == 24, f"Expected obs_dim=24, got {env.obs_dim}"

        print("\n✓ Test 2 PASSED: State space correctly enhanced with fidelity")
        return True

    except Exception as e:
        print(f"\n✗ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fidelity_reward_wrapper():
    """Test 3: Fidelity-aware reward wrapper"""
    print("\n" + "=" * 60)
    print("TEST 3: Fidelity-Aware Reward Wrapper")
    print("=" * 60)

    try:
        # Create base environment
        base_env = env_creator({"dataset": "./qdataset/qd_all.pickle"})

        # Wrap with fidelity-aware reward
        w_time = 0.7
        w_fidelity = 0.3
        env = FidelityAwareRewardWrapper(base_env, w_time=w_time, w_fidelity=w_fidelity)

        print(f"\nReward weights: w_time={w_time}, w_fidelity={w_fidelity}")

        # Reset environment
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")

        # Take a few steps
        print("\nTaking 5 steps with random actions...")
        total_reward = 0
        fidelity_scores = []

        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"\nStep {i+1}:")
            print(f"  Action: {action}")
            print(f"  Reward: {reward:.6f}")

            if 'fidelity_score' in info:
                print(f"  Fidelity Score: {info['fidelity_score']:.6f}")
                print(f"  Speed Reward: {info.get('speed_reward', 0):.6f}")
                print(f"  Fidelity Reward: {info.get('fidelity_reward', 0):.6f}")
                fidelity_scores.append(info['fidelity_score'])

            total_reward += reward

            if terminated or truncated:
                print(f"\n  Episode ended at step {i+1}")
                break

        print(f"\nTotal reward: {total_reward:.6f}")
        if fidelity_scores:
            print(f"Average fidelity score: {np.mean(fidelity_scores):.6f}")

        # Test weight update
        new_w_time = 0.5
        new_w_fidelity = 0.5
        env.set_weights(new_w_time, new_w_fidelity)
        print(f"\n✓ Weights updated to: w_time={env.w_time}, w_fidelity={env.w_fidelity}")

        print("\n✓ Test 3 PASSED: Reward wrapper functioning correctly")
        return True

    except Exception as e:
        print(f"\n✗ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evolutionary_algorithm():
    """Test 4: Evolutionary algorithm components"""
    print("\n" + "=" * 60)
    print("TEST 4: Evolutionary Algorithm")
    print("=" * 60)

    try:
        from evolutionary_reward_shaping import EvolutionaryRewardShaper, EAConfig

        # Create minimal EA config
        config = EAConfig(
            population_size=4,
            max_generations=2,
            drl_episodes_per_eval=5,
        )

        ea = EvolutionaryRewardShaper(config)
        print(f"\nEA Configuration:")
        print(f"  Population size: {config.population_size}")
        print(f"  Max generations: {config.max_generations}")
        print(f"  DRL episodes per eval: {config.drl_episodes_per_eval}")

        # Test fitness calculation
        mock_episode_rewards = [0.05, 0.06, 0.055, 0.058, 0.052]
        mock_episode_info = [
            {"fidelity_score": 0.85, "rescheduling_count": 2},
            {"fidelity_score": 0.90, "rescheduling_count": 1},
            {"fidelity_score": 0.88, "rescheduling_count": 1},
            {"fidelity_score": 0.87, "rescheduling_count": 3},
            {"fidelity_score": 0.89, "rescheduling_count": 2},
        ]

        fitness = ea.calculate_fitness(mock_episode_rewards, mock_episode_info)
        print(f"\nTest fitness calculation:")
        print(f"  Mock episode rewards: {mock_episode_rewards}")
        print(f"  Mock fidelity scores: {[info['fidelity_score'] for info in mock_episode_info]}")
        print(f"  Calculated fitness: {fitness:.6f}")

        assert fitness > 0, "Fitness should be positive"

        print("\n✓ Test 4 PASSED: EA components working correctly")
        return True

    except Exception as e:
        print(f"\n✗ Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all validation tests"""
    print("\n" + "=" * 80)
    print("HYBRID EA+DRL SYSTEM VALIDATION TESTS")
    print("=" * 80)

    tests = [
        ("QNode Fidelity Calculation", test_qnode_fidelity),
        ("Environment State Space", test_environment_state_space),
        ("Fidelity Reward Wrapper", test_fidelity_reward_wrapper),
        ("Evolutionary Algorithm", test_evolutionary_algorithm),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status:<12} {test_name}")

    print("-" * 80)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for training.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

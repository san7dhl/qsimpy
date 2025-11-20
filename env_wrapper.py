import gymnasium as gym
from gymnasium.core import Env
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward
import numpy as np
from gymnasium.spaces import Box


class ScaleQSimPyEnv(gym.RewardWrapper):
    def __init__(self, env: Env, scale: float):
        super().__init__(env)
        self.scaling_factor = scale

    def reward(self, reward):
        reward *= self.scaling_factor
        return reward


class FidelityAwareRewardWrapper(gym.Wrapper):
    """
    Fidelity-Aware Reward Wrapper for Hybrid EA+DRL System.

    Modifies the reward function to include fidelity penalties based on:
    - Speed reward (time efficiency)
    - Fidelity reward (success probability)

    Reward formula:
    r_t = (w_time / time) + w_fidelity * log(P_success) - Penalties

    Args:
        env: The base QSimPy environment
        w_time: Weight for time-based reward (default: 0.7)
        w_fidelity: Weight for fidelity-based reward (default: 0.3)
    """

    def __init__(self, env: Env, w_time: float = 0.7, w_fidelity: float = 0.3):
        super().__init__(env)
        self.w_time = w_time
        self.w_fidelity = w_fidelity
        self.last_action = None
        self.last_fidelity = 1.0

    def step(self, action):
        # Store action to retrieve fidelity after step
        self.last_action = action

        # Execute the original step
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Modify reward to include fidelity component
        # Original reward is 1/time (for successful tasks) or -0.1 (for failed constraints)
        if reward > 0:  # Successful task submission
            # Get fidelity of the selected node for the current task
            qnode = self.env.qnodes[action]
            current_task = self.env.prev_qtask if self.env.prev_qtask else self.env.current_qtask

            if current_task is not None:
                fidelity_score = qnode.get_fidelity_score(current_task)
                self.last_fidelity = fidelity_score

                # Fidelity reward: log(P_success)
                # Use log to penalize low fidelity more heavily
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                fidelity_reward = np.log(fidelity_score + epsilon)

                # Combined reward: speed + fidelity
                # reward is already 1/time, so we keep it as speed_reward
                speed_reward = reward
                modified_reward = self.w_time * speed_reward + self.w_fidelity * fidelity_reward

                # Store components in info for tracking
                info['speed_reward'] = speed_reward
                info['fidelity_reward'] = fidelity_reward
                info['fidelity_score'] = fidelity_score
                info['w_time'] = self.w_time
                info['w_fidelity'] = self.w_fidelity

                return obs, modified_reward, terminated, truncated, info

        # For constraint violations or other cases, return original reward
        return obs, reward, terminated, truncated, info

    def set_weights(self, w_time: float, w_fidelity: float):
        """Update reward weights (used by EA during evolution)."""
        self.w_time = w_time
        self.w_fidelity = w_fidelity


class GymNormalizeObservation(NormalizeObservation):
    def __init__(self, env: Env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.observation_space = Box(
            low=np.ones((self.env.obs_dim,)) * -np.inf,
            high=np.ones((self.env.obs_dim,)) * np.inf,
        )

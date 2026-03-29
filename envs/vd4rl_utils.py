import os
import numpy as np
import gymnasium as gym
from torchrl.data.datasets import VD4RLExperienceReplay

from utils.datasets import Dataset


class DMCToGymWrapper(gym.Wrapper):
    """Wrap a dm_env-style DMC environment to a Gymnasium-compatible interface."""

    def __init__(self, env):
        self._env = env
        self.obs_spec = env.observation_spec()
        self.act_spec = env.action_spec()

    @property
    def unwrapped(self):
        return self._env

    def reset(self, **kwargs):
        time_step = self._env.reset()
        return np.transpose(time_step.observation, axes=(1,2,0)), {}

    def step(self, action):
        time_step = self._env.step(action)
        terminated = time_step.last()
        truncated = False
        return np.transpose(time_step.observation, axes=(1,2,0)), time_step.reward, terminated, truncated, {}

    def render(self):
        # Render from the underlying physics engine
        return self._env._env._env._env._env.physics.render(height=256, width=256, camera_id=0)

    def __getattr__(self, name):
        return getattr(self._env, name)


def get_dataset(env_name, dataset_root=None):
    if dataset_root is not None:
        dataset_root = os.path.expanduser(dataset_root)
    vd4rl = VD4RLExperienceReplay(
        env_name,
        root=dataset_root,
        batch_size=32,
    )

    # 2. Access the full underlying TensorDict
    td = vd4rl._storage._storage

    # 3. Convert to numpy arrays
    observations = td['pixels'].numpy()            # [N, 64, 64, 3] uint8
    next_observations = td['next', 'pixels'].numpy()  # [N, 64, 64, 3] uint8
    actions = td['action'].numpy().astype(np.float32)  # [N, act_dim] float32
    rewards = td['next', 'reward'].numpy().astype(np.float32).squeeze(-1)  # [N]

    # terminals: episode ended (either terminated or truncated)
    terminals = td['next', 'done'].numpy().astype(np.float32).squeeze(-1)  # [N]
    masks = 1.0 - terminals  # [N]

    # 4. Create Dataset
    dataset = Dataset.create(
        observations=observations,
        next_observations=next_observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        masks=masks,
    )

    return dataset
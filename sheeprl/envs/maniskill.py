import gymnasium as gym
import numpy as np
import torch
import mani_skill.envs  # registers environments

class ManiSkillWrapper(gym.Wrapper):
    def __init__(
        self,
        env_id="PushCube-v1",
        num_envs=1,                # Number of parallel environments
        obs_mode="state",                # "state" or "rgb"
        control_mode="pd_ee_delta_pos",  # or "pd_joint_delta_pos"
        render_mode="rgb_array",         # Passed to make(), not used in render()
        reward_mode="dense",             # "sparse", "dense", etc.
        record_video=False,
    ):
        self.obs_mode = obs_mode
        self.record_video = record_video

        # Create underlying ManiSkill env
        env = gym.make(
            env_id,
            obs_mode=obs_mode,
            num_envs=num_envs,
            control_mode=control_mode,
            render_mode=render_mode,
            reward_mode=reward_mode,
        )
        super().__init__(env)

        # Determine and normalize observation space
        if obs_mode == "rgb":
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(3, 128, 128),
                dtype=np.uint8
            )

        elif isinstance(env.observation_space, gym.spaces.Box):
            shape = env.observation_space.shape
            if len(shape) == 2 and shape[0] == 1:
                shape = (shape[1],)

            self.observation_space = gym.spaces.Box(
                low=np.squeeze(env.observation_space.low),
                high=np.squeeze(env.observation_space.high),
                shape=shape,
                dtype=env.observation_space.dtype,
            )

        else:
            raise ValueError("Unsupported observation space format")

        self.action_space = env.action_space
        self.reward_range = (-np.inf, np.inf)
        self._metadata = {"render_fps": 20}

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def _process_obs(self, obs):
        if self.obs_mode == "state":
            obs = self._to_numpy(obs)
            obs = np.squeeze(obs)
            return obs

        elif self.obs_mode == "rgb":
            rgb = obs.get("sensor_data", {}).get("base_camera", {}).get("rgb", None)
            if rgb is None:
                raise ValueError("Missing sensor_data['base_camera']['rgb'] in observation")

            rgb = self._to_numpy(rgb)

            # Remove batch dimension only
            if rgb.ndim == 4 and rgb.shape[0] == 1:
                rgb = rgb[0]

            # Transpose if needed (ensure CHW)
            if rgb.ndim == 3 and rgb.shape[-1] == 3:
                rgb = np.transpose(rgb, (2, 0, 1))

            # Final sanity check
            if rgb.shape != (3, 128, 128):
                raise ValueError(f"Unexpected RGB shape: {rgb.shape}")

            return rgb


    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._process_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._process_obs(obs)

        reward = self._to_numpy(reward).item()
        terminated = bool(self._to_numpy(terminated))
        truncated = bool(self._to_numpy(truncated))

        return obs, reward, terminated, truncated, info

    def render(self):
        frame = self.env.render()

        # Convert tensor to numpy
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()

        # Remove batch dimension
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]

        # Convert CHW → HWC
        if frame.ndim == 3 and frame.shape[0] in [1, 3]:
            frame = np.transpose(frame, (1, 2, 0))

        # Convert to uint8
        if frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)

        return frame

    def close(self):
        self.env.close()

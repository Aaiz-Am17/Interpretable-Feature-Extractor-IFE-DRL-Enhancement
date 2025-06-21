"""
Fix for the environment wrappers to use Gymnasium instead of retro
"""
import os
import cv2
import gymnasium as gym
import numpy as np
from collections import deque

# Custom FrameStack implementation
class FrameStack(gym.Wrapper):
    """Stack k last frames.
    
    Returns lazy array, which is much more memory efficient.
    """
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shp[0] * k, *shp[1:]),
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
    # Make sure we stack along the channel dimension (first dimension)
        frames_array = np.array(list(self.frames))
        return frames_array.reshape(-1, *frames_array.shape[2:])


def create_env(args, decorr_steps=None):
    """Create a wrapped environment for Atari games"""
    env_name = args.env_name
    
    # Handle Atari environments
    if 'NoFrameskip' in env_name:
        # Create vectorized environments
        envs = []
        for i in range(args.parallel_envs):
            # Create the base environment
            env = gym.make(env_name, render_mode=None)
            
            # Apply wrappers
            env = gym.wrappers.AtariPreprocessing(
                env,
                noop_max=30,
                frame_skip=args.frame_skip,
                screen_size=84,  # Make sure this is 84x84
                terminal_on_life_loss=False,
                grayscale_obs=True,
                grayscale_newaxis=True,  # Change to True to ensure proper channel dimension
                scale_obs=False
            )
            env = FrameStack(env, 4)
            
            # Decorrelate environments if needed
            if decorr_steps is not None and decorr_steps > 0:
                obs, _ = env.reset()
                for _ in range(i * decorr_steps):
                    action = env.action_space.sample()
                    obs, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        obs, _ = env.reset()
            
            envs.append(env)
        
        # Create a simple wrapper to handle parallel environments
        return ParallelEnvWrapper(envs)
    else:
        raise ValueError(f"Environment {env_name} not supported. Please use an Atari NoFrameskip environment.")

class ParallelEnvWrapper:
    """Wrapper for parallel environments"""
    def __init__(self, envs):
        self.envs = envs
        self.num_envs = len(envs)
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        self._async_actions = None
    
    def reset(self):
        """Reset all environments"""
        observations = []
        for env in self.envs:
            obs, _ = env.reset()
            observations.append(obs)
        return np.array(observations)
    
    def step_async(self, actions):
        """Store actions for async execution"""
        self._async_actions = actions
    
    def step_wait(self):
        """Execute stored actions and wait for results"""
        results = [env.step(a) for env, a in zip(self.envs, self._async_actions)]
        obs, rewards, terminateds, truncateds, infos = zip(*results)
        
        # Convert to numpy arrays
        obs = np.array(obs)
        rewards = np.array(rewards)
        dones = np.array([t or tr for t, tr in zip(terminateds, truncateds)])
        
        # Process infos
        processed_infos = []
        for i, (terminated, truncated, info) in enumerate(zip(terminateds, truncateds, infos)):
            # Add episode metrics if episode is done
            if terminated or truncated:
                if not 'episode_metrics' in info:
                    info['episode_metrics'] = {
                        'return': 0,
                        'length': 0,
                        'time': 0,
                        'discounted_return': 0
                    }
            processed_infos.append(info)
        
        return obs, rewards, dones, processed_infos
    
    # Add this method
    def step(self, actions):
        """Execute actions and return results"""
        self.step_async(actions)
        return self.step_wait()
    
    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()
# gym_adapter.py
import gymnasium as gym
import numpy as np

class GymAdapter:
    """Adapter class to make Gymnasium work with code designed for old Gym API"""
    
    @staticmethod
    def make(env_name):
        """Create a Gymnasium environment that works like old Gym"""
        if 'NoFrameskip' in env_name:
            # Convert old Gym env name to Gymnasium format
            env_name = env_name.replace('NoFrameskip-v4', '-v5')
        
        env = gym.make(env_name, render_mode=None)
        return GymEnvWrapper(env)

class GymEnvWrapper:
    """Wrapper to adapt Gymnasium to old Gym interface"""
    
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        
    def reset(self):
        obs, _ = self.env.reset()
        return obs
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
    
    def seed(self, seed=None):
        if seed is not None:
            self.env.reset(seed=seed)
        return [seed]
        
    def render(self):
        return self.env.render()
        
    def close(self):
        return self.env.close()
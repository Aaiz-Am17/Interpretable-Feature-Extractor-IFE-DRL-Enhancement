"""
This file includes the model and environment setup and the main training loop.
Look at the README.md file for details on how to use this.
"""

import time, random
from collections import deque
from pathlib import Path
from types import SimpleNamespace as sn
import os
import argparse
import torch, wandb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from tqdm import trange
from rich import print

# Replace these imports
# from common import argp
# from common.rainbow import Rainbow
# from common.env_wrappers import create_env, BASE_FPS_ATARI, BASE_FPS_PROCGEN

# With these imports
import argparse
from fix_env import create_env
from networks import ImpalaCNNLarge, NatureCNN, DuelingNatureCNN, ImpalaCNNSmall, Dueling
BASE_FPS_ATARI = 60.0
BASE_FPS_PROCGEN = 15.0

torch.backends.cudnn.benchmark = True  # let cudnn heuristics choose fastest conv algorithm

class LinearSchedule:
    """Linear interpolation between initial_value and final_value over
    schedule_timesteps. After this many timesteps pass final_value is returned.
    """
    def __init__(self, initial_step, initial_value, final_value, decay_time):
        """
        Parameters
        ----------
        initial_step: int
            Initial timestep for the schedule. Typically should be set to 0.
        initial_value: float
            Initial value that is returned by the schedule at the start of the decay.
        final_value: float
            Final value that is returned by the schedule after decay_time steps.
        decay_time: int
            Number of timesteps over which the schedule should decay
            from initial_value to final_value.
        """
        self.initial_step = initial_step
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_time = decay_time

    def __call__(self, step):
        """Returns the scheduled value."""
        decay_progress = min(max(0.0, (step - self.initial_step) / self.decay_time), 1.0)
        return self.initial_value + (self.final_value - self.initial_value) * decay_progress
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', dest='env_name', type=str, default='PongNoFrameskip-v4', help='environment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--parallel_envs', type=int, default=8, help='number of parallel environments')
    parser.add_argument('--frame_skip', type=int, default=4, help='number of frames to skip')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb for logging')
    parser.add_argument('--force_cpu', action='store_true', help='force CPU usage')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='directory to save models')
    
    # Model specific args
    parser.add_argument('--model', type=str, default='impala_large:1', help='model architecture')
    parser.add_argument('--spectral_norm', default=False, help='use spectral normalization')
    
    # PPO specific args
    parser.add_argument('--ppo_steps', type=int, default=2048, help='steps per PPO update')
    parser.add_argument('--batch_size', type=int, default=64, help='minibatch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--lambda_gae', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_ratio', type=float, default=0.2, help='PPO clip ratio')
    parser.add_argument('--value_coef', type=float, default=0.5, help='value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--training_frames', type=int, default=10000000, help='total training frames')
    parser.add_argument('--ppo_epochs', type=int, default=4, help='number of PPO epochs')
    parser.add_argument('--grad_norm_max', type=float, default=0.5, help='max gradient norm')
    
    # Exploration parameters (for compatibility)
    parser.add_argument('--init_eps', type=float, default=1.0, help='initial epsilon for exploration')
    parser.add_argument('--final_eps', type=float, default=0.01, help='final epsilon for exploration')
    parser.add_argument('--eps_decay_frames', type=int, default=1000000, help='frames over which to decay epsilon')
    
    # Optional wandb tag
    parser.add_argument('--wandb_tag', type=str, default=None, help='tag for wandb run')
    
    args = parser.parse_args()
    return args, vars(args)
# PPO Buffer for experience collection
class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size,) + obs_dim, dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.int32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.burnedin = False
        
    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
        
        # Mark as burnedin once we have enough data
        if self.ptr >= self.max_size:
            self.burnedin = True
        
    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
        
        # Rewards-to-go
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr
        
    def get(self):
        assert self.ptr == self.max_size  # Buffer must be full before using
        self.ptr, self.path_start_idx = 0, 0
        
        # Normalize advantages
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        
        return dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf
        )
    
    def _discount_cumsum(self, x, discount):
        discounted_sum = np.zeros_like(x)
        discounted_sum[-1] = x[-1]
        for t in reversed(range(len(x)-1)):
            discounted_sum[t] = x[t] + discount * discounted_sum[t+1]
        return discounted_sum

# PPO Implementation that replaces Rainbow
class PPO:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        
        # Create the model
        # from common.model_registry import get_model  # Remove, we create model directly now
        # model_cls = get_model(args.model, spectral_norm=args.spectral_norm)
        in_depth = env.observation_space.shape[0]
        
        self.policy = ImpalaCNNLarge(in_depth, env.action_space.n, 
                               nn.Linear, spectral_norm=False) # Use the ImpalaCNNLarge directly

        if torch.cuda.is_available() and not args.force_cpu:
            self.device = torch.device('cuda')
            self.policy = self.policy.to(self.device)
        else:
            self.device = torch.device('cpu')
            
        # Configure optimizer
        self.opt = optim.Adam(self.policy.parameters(), lr=args.lr)
        
        # Create PPO buffer
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.n
        self.buffer = PPOBuffer(obs_dim, act_dim, args.ppo_steps, args.gamma, args.lambda_gae)
        
        # PPO hyperparameters
        self.clip_ratio = args.clip_ratio
        self.value_coef = args.value_coef
        self.entropy_coef = args.entropy_coef
        self.ppo_epochs = args.ppo_epochs
        self.rewards = []
        self.value = 0
        
    def act(self, states, explore=True):
        """
        Get actions from policy
        """
        states = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            action_probs, values, _ = self.forward(states)
            if explore:
                dist = Categorical(action_probs)
                actions = dist.sample().cpu().numpy()
                log_probs = dist.log_prob(torch.tensor(actions, device=self.device)).cpu().numpy()
            else:
                actions = action_probs.argmax(dim=1).cpu().numpy()
                log_probs = np.log(action_probs.cpu().numpy()[np.arange(len(actions)), actions])
            self.value = values.cpu().numpy()
        
        return actions, values.cpu().numpy(), log_probs
    
    def forward(self, x):
        """
        Forward pass through the model, extracting policy, value, and attention
        """
        print(f"Input shape in forward: {x.shape}")
        x = x / 255.0  # Normalize
        q_values = self.policy(x)
        
        # Extract attention masks for visualization - Modified
        attention = None
        if hasattr(self.policy, 'HUE') and hasattr(self.policy.HUE, 'att'):
            attention = self.policy.HUE.att
        
        # For dueling architecture, we need to handle differently
        if isinstance(q_values, tuple):
            action_probs = F.softmax(q_values[0], dim=1)
            value = q_values[1]
        else:
            action_probs = F.softmax(q_values, dim=1)
            value = torch.mean(q_values, dim=1, keepdim=True)
            
        return action_probs, value, attention
    
    def train(self, batch_size, clip_ratio=0.2):
        """
        Train the model using PPO
        """
        data = self.buffer.get()
        
        # Convert to tensors
        obs = torch.FloatTensor(data['obs']).to(self.device)
        act = torch.LongTensor(data['act']).to(self.device)
        ret = torch.FloatTensor(data['ret']).to(self.device)
        adv = torch.FloatTensor(data['adv']).to(self.device)
        old_logp = torch.FloatTensor(data['logp']).to(self.device)
        
        total_loss = 0
        value_loss = 0
        policy_loss = 0
        entropy = 0
        clip_frac = 0
        
        # Training loop
        for _ in range(self.ppo_epochs):
            # Generate random indices for mini-batches
            indices = torch.randperm(len(obs))
            
            # Process in mini-batches
            for start in range(0, len(obs), batch_size):
                end = start + batch_size
                if end > len(obs):
                    end = len(obs)
                
                batch_indices = indices[start:end]
                
                # Get mini-batch data
                obs_batch = obs[batch_indices]
                act_batch = act[batch_indices]
                adv_batch = adv[batch_indices]
                ret_batch = ret[batch_indices]
                old_logp_batch = old_logp[batch_indices]
                
                # Forward pass
                action_probs, values, _ = self.forward(obs_batch)
                values = values.squeeze(-1)
                
                # Calculate log probabilities of actions
                dist = Categorical(action_probs)
                new_logp_batch = dist.log_prob(act_batch)
                entropy_batch = dist.entropy().mean()
                
                # Calculate ratios and surrogate objectives for PPO
                ratio = torch.exp(new_logp_batch - old_logp_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_batch
                policy_loss_batch = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss_batch = F.mse_loss(values, ret_batch)
                
                # Total loss
                loss = policy_loss_batch + self.value_coef * value_loss_batch - self.entropy_coef * entropy_batch
                
                # Backpropagation
                self.opt.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.grad_norm_max)
                
                self.opt.step()
                
                # Track metrics
                total_loss += loss.item()
                policy_loss += policy_loss_batch.item()
                value_loss += value_loss_batch.item()
                entropy += entropy_batch.item()
                clip_frac += (abs(ratio - 1.0) > clip_ratio).float().mean().item()
        
        # Calculate average metrics
        num_updates = self.ppo_epochs * (len(obs) // batch_size + 1)
        avg_loss = total_loss / num_updates
        avg_policy_loss = policy_loss / num_updates
        avg_value_loss = value_loss / num_updates
        avg_entropy = entropy / num_updates
        avg_clip_frac = clip_frac / num_updates
        
        # Get gradient norm
        grad_norm = 0
        for p in self.policy.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5
        
        return avg_loss, avg_policy_loss, avg_value_loss, avg_entropy, avg_clip_frac, grad_norm
    
    def save(self, frame_idx, args=None, run_name=None, run_id=None, target_metric=None, returns_all=None, q_values_all=None):
        """
        Save policy to disk
        """
        state = {
            'frame_idx': frame_idx,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
        }
        
        if args is not None:
            state['args'] = vars(args)
        if target_metric is not None:
            state['metric'] = target_metric
        if run_name is not None:
            state['wandb_run_name'] = run_name
        if run_id is not None:
            state['wandb_run_id'] = run_id
        if returns_all is not None:
            state['returns_all'] = returns_all
        if q_values_all is not None:
            state['q_values_all'] = q_values_all
            
        path = os.path.join(args.save_dir, f"model_{frame_idx}.pt")
        torch.save(state, path)
        
        # Also save to wandb
        if args.use_wandb:
            wandb.save(path)

if __name__ == '__main__':
    # Add PPO-specific arguments
    argp = argparse.ArgumentParser()
    argp.add_argument('--env', dest='env_name', type=str, default='PongNoFrameskip-v4', help='environment name')
    argp.add_argument('--seed', type=int, default=0, help='random seed')
    argp.add_argument('--parallel_envs', type=int, default=8, help='number of parallel environments')
    argp.add_argument('--frame_skip', type=int, default=4, help='number of frames to skip')
    argp.add_argument('--use_wandb', action='store_true', help='use wandb for logging')
    argp.add_argument('--force_cpu', action='store_true', help='force CPU usage')
    argp.add_argument('--save_dir', type=str, default='./checkpoints', help='directory to save models')
    
    # PPO specific args
    argp.add_argument('--ppo_steps', type=int, default=2048, help='steps per PPO update')
    argp.add_argument('--batch_size', type=int, default=64, help='minibatch size')
    argp.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    argp.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    argp.add_argument('--lambda_gae', type=float, default=0.95, help='GAE lambda')
    argp.add_argument('--clip_ratio', type=float, default=0.2, help='PPO clip ratio')
    argp.add_argument('--value_coef', type=float, default=0.5, help='value loss coefficient')
    argp.add_argument('--entropy_coef', type=float, default=0.01, help='entropy coefficient')
    argp.add_argument('--training_frames', type=int, default=10000000, help='total training frames')
    
    args, wandb_log_config = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set up logging & model checkpoints
    name = args.env_name.replace(':','-')
    watermark = f"{name}_ppo_ife"
    wandb.init(project=f'ppo_ife_{name}', save_code=True, config=dict(
        env_name=args.env_name,
        seed=args.seed,
        parallel_envs=args.parallel_envs,
        frame_skip=args.frame_skip,
        use_wandb=args.use_wandb,
        force_cpu=args.force_cpu,
        save_dir=args.save_dir,
        ppo_steps=args.ppo_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        lambda_gae=args.lambda_gae,
        clip_ratio=args.clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        training_frames=args.training_frames,
        log_version=100
    ), name=watermark,
               mode=('online' if args.use_wandb else 'offline'), anonymous='allow', tags=[args.wandb_tag] if args.wandb_tag else [])
    save_dir = Path("checkpoints") / wandb.run.name
    save_dir.mkdir(parents=True, exist_ok=True)
    args.save_dir = str(save_dir)

    # create exploration schedule
    eps_schedule = LinearSchedule(0, initial_value=args.init_eps, final_value=args.final_eps, decay_time=args.eps_decay_frames)

    print(f'Creating {args.parallel_envs} environment instances. This may take up to a few minutes.. ', end='')
    env = create_env(args, decorr_steps=None)
    states = env.reset()
    print('Done.')

    # Initialize PPO agent (replacing Rainbow)
    ppo_agent = PPO(env, args)
    wandb.watch(ppo_agent.policy)

    print('[blue bold]Running environment =', args.env_name,
          '[blue bold]\nwith action space   =', env.action_space,
          '[blue bold]\nobservation space   =', env.observation_space)

    episode_count = 0
    returns = deque(maxlen=100)
    discounted_returns = deque(maxlen=10)
    losses = deque(maxlen=10)
    policy_losses = deque(maxlen=10)
    value_losses = deque(maxlen=10)
    entropies = deque(maxlen=10)
    clip_fracs = deque(maxlen=10)
    grad_norms = deque(maxlen=10)
    iter_times = deque(maxlen=10)
    reward_density = 0

    returns_all = []
    
    # Record attention visualizations
    def visualize_attention(states, attention, save_path):
        """Visualize attention masks"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        
        # Original frame
        plt.subplot(1, 2, 1)
        plt.imshow(states[0][0], cmap='gray')
        plt.title('Input Frame')
        plt.axis('off')
        
        # Attention overlay
        plt.subplot(1, 2, 2)
        plt.imshow(states[0][0], cmap='gray')
        plt.imshow(attention[0].cpu().numpy(), cmap='hot', alpha=0.5)
        plt.title('Attention Mask')
        plt.axis('off')
        
        plt.savefig(save_path)
        plt.close()

    # main training loop
    # Modified to use PPO instead of Rainbow
    steps_per_update = args.ppo_steps // args.parallel_envs
    t = trange(0, args.training_frames + 1, args.parallel_envs)
    for game_frame in t:
        iter_start = time.time()
        
        # Reset buffer when full
        if ppo_agent.buffer.ptr >= args.ppo_steps:
            # Train PPO
            loss, policy_loss, value_loss, entropy, clip_frac, grad_norm = ppo_agent.train(args.batch_size, clip_ratio=args.clip_ratio)
            losses.append(loss)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropies.append(entropy)
            clip_fracs.append(clip_frac)
            grad_norms.append(grad_norm)

        # Get actions using current policy
        eps = eps_schedule(game_frame)
        explore = np.random.random() < eps  # Use epsilon-greedy
        actions, values, log_probs = ppo_agent.act(states, explore=explore)

        # Take actions in environment
        next_states, rewards, dones, infos = env.step(actions)
        
        # Store transitions in PPO buffer
        for i in range(args.parallel_envs):
            ppo_agent.buffer.store(states[i], actions[i], rewards[i], values[i], log_probs[i])
            
            # Handle episode completion
            if dones[i]:
                # Finish path in buffer
                ppo_agent.buffer.finish_path()
                
                # Log episode stats if available
                if 'episode_metrics' in infos[i]:
                    episode_metrics = infos[i]['episode_metrics']
                    returns.append(episode_metrics['return'])
                    returns_all.append((game_frame, episode_metrics['return']))
                    discounted_returns.append(episode_metrics['discounted_return'])
                    
                    log = {
                        'x/game_frame': game_frame + i, 
                        'x/episode': episode_count,
                        'ep/return': episode_metrics['return'],
                        'ep/length': episode_metrics['length'],
                        'ep/time': episode_metrics['time'],
                        'ep/mean_reward_per_frame': episode_metrics['return'] / (episode_metrics['length'] + 1),
                        'grad_norm': np.mean(grad_norms) if grad_norms else 0,
                        'mean_loss': np.mean(losses) if losses else 0,
                        'mean_policy_loss': np.mean(policy_losses) if policy_losses else 0,
                        'mean_value_loss': np.mean(value_losses) if value_losses else 0,
                        'mean_entropy': np.mean(entropies) if entropies else 0,
                        'mean_clip_frac': np.mean(clip_fracs) if clip_fracs else 0,
                        'fps': args.parallel_envs / np.mean(iter_times) if iter_times else 0,
                        'running_avg_return': np.mean(returns),
                        'reward_density': reward_density,
                        'epsilon': eps
                    }
                    
                    # Log video recordings if available
                    if 'emulator_recording' in infos[i]:
                        log['emulator_recording'] = wandb.Video(
                            infos[i]['emulator_recording'],
                            fps=(BASE_FPS_PROCGEN if args.env_name.startswith('procgen:') else BASE_FPS_ATARI),
                            format="mp4"
                        )
                    if 'preproc_recording' in infos[i]:
                        log['preproc_recording'] = wandb.Video(
                            infos[i]['preproc_recording'],
                            fps=(BASE_FPS_PROCGEN if args.env_name.startswith('procgen:') else BASE_FPS_ATARI) // args.frame_skip,
                            format="mp4"
                        )
                    
                    wandb.log(log)
                    episode_count += 1
            
            elif game_frame % steps_per_update == 0:
                # For non-terminal steps, estimate value of next state
                with torch.no_grad():
                    next_states_tensor = torch.FloatTensor(np.array([next_states[i]])).to(ppo_agent.device)
                    _, next_value, _ = ppo_agent.forward(next_states_tensor)
                    next_value = next_value.cpu().numpy()[0][0]
                ppo_agent.buffer.finish_path(next_value)
        
        # Track reward density
        reward_density = 0.999 * reward_density + 0.001 * np.mean(rewards != 0)
        
        # Update state
        states = next_states
        
        # Visualize attention periodically
        if game_frame % 100000 == 0 and hasattr(ppo_agent.policy, 'HUE') and hasattr(ppo_agent.policy.HUE, 'att'):
            with torch.no_grad():
                states_tensor = torch.FloatTensor(states).to(ppo_agent.device)
                _, _, attention = ppo_agent.forward(states_tensor)
                attention_path = os.path.join(args.save_dir, f"attention_{game_frame}.png")
                visualize_attention(states, attention, attention_path)
                wandb.log({'attention_visualization': wandb.Image(attention_path)})
        
        if game_frame % (50_000-(50_000 % args.parallel_envs)) == 0:
            print(f' [{game_frame:>8} frames, {episode_count:>5} episodes] running average return = {np.mean(returns)}')
            torch.cuda.empty_cache()
        
        # Save model checkpoint
        if game_frame % (5_000_000-(5_000_000 % args.parallel_envs)) == 0 and game_frame > 0:
            ppo_agent.save(game_frame, args=args, run_name=wandb.run.name, run_id=wandb.run.id, 
                          target_metric=np.mean(returns), returns_all=returns_all)
            print(f'Model saved at {game_frame} frames.')
        
        iter_times.append(time.time() - iter_start)
        t.set_description(f' [{game_frame:>8} frames, {episode_count:>5} episodes]', refresh=False)
    
    wandb.log({'x/game_frame': game_frame + args.parallel_envs, 'x/episode': episode_count})
    env.close()
    wandb.finish()
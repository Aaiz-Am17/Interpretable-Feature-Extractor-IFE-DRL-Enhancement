import time, random
from collections import deque
from pathlib import Path
from types import SimpleNamespace as sn
# For ROMs, use ale-py
from ale_py import ALEInterface
#ale = ALEInterface()
import torch, wandb
import numpy as np
from tqdm import trange
from rich import print

from common import argp
from common.test_rainbow import Rainbow
from common.test_env_wrappers import create_atari_test_env, BASE_FPS_ATARI, BASE_FPS_PROCGEN
from common.utils import LinearSchedule, get_mean_ep_length
import cv2
# import scipy.ndimage
torch.backends.cudnn.benchmark = True  # let cudnn heuristics choose fastest conv algorithm




if __name__ == '__main__':
    args, wandb_log_config = argp.read_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    name = args.env_name.replace(":","-")
    case = Path(f"standard")
    save_dir = Path(f"video/{case}/")
    args.save_dir = str(save_dir)
    args.test = True
    env = create_atari_test_env(args)
    
    states = env.reset()
    rainbow = Rainbow(env, args)
    path = f"checkpoints/{name}_{case}/checkpoint_49999680.pt"
    rainbow.load(path)

    step = 0
    while 1:
        action = rainbow.act(states, 0)
        # Combine the attention mask with the action to visualize it in a record wrapper
        action_combined = [action, rainbow.q_policy.HUE.att.cpu()]

        step+=1
        states, reward, done, info = env.step(action_combined)

        if done and not env.was_real_done:
            states = env.reset()
        elif env.was_real_done:
            break


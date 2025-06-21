# Pay Attention to What and Where? Interpretable Feature Extractor for Vision-based Deep Reinforcement Learning

This repository is the official implementation of [Pay Attention to What and Where? Interpretable Feature Extractor for Vision-based Deep Reinforcement Learning] for submission in IJCNN 2025

## Proposed approach

![Alt text](github_asset/outline.png?raw=true "Proposed Approach for Interpretable Feature Extractor")

## Requirements

While the ATARI environments for the project is mainly used for evaluation. We also aim to support `gym-retro` and other environment as in [FastRainbow](https://github.com/schmidtdominik/Rainbow). Therefore, make sure your environment are using `python=3.7`.

Install necessary prerequisites with

```
sudo apt install zlib1g-dev cmake unrar
pip install wandb gym[atari]==0.18.0 imageio moviepy torchsummary tqdm rich procgen gym-retro torch stable_baselines3 atari_py==0.2.9 shimmy>=0.2.1
```

Or you can only need to install all the packages by

```
pip install -r requirements.txt
```

If you intend to use `gym` Atari games, you will need to install these ROMs separately, e.g., by running:

```
wget http://www.atarimania.com/roms/Roms.rar
unrar x Roms.rar
python -m atari_py.import_roms .
```

To set up `gym-retro` games you should follow the instructions [here](https://retro.readthedocs.io/en/latest/getting_started.html#importing-roms).

This environment set up is tested in Ubuntu 22.04. We experience some errors (mainly unsupported package in orther Linux Distributions such as CentOS Linux release 7.9.2009 or Red Hat Enterprise Linux 7). Please aware the different in the Linux Distribution and modify it accordingly.

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --env_name gym:Boxing
```

This command will start the training script on Boxing game with default hyperparameters (see `common/argp.py` for more details),and log all results to "Weights and Biases" and the checkpoints directory. We name the checkpoint folder with the similar name as the wandb watermark. You should check if the name exits in the checkpoints folder (We don't want to erase it automatically, it's paintful)

## Pre-trained Models

We provide the checkpoints of Boxing, Assault and SpaceInvaders games stored in this [link](https://drive.google.com/drive/folders/1SMsP_MQN5sxevRQQ5ZmnklfT3o_C5tgQ?usp=drive_link), please download it for the evaluation and leave it in the checkpoints folder. The remaining could also be provided, please contact the author for detail.

## Evaluation

To evaluate the model on the game Boxing, run:

```eval
python eval.py --env_name gym:Boxing
```

The script will load the model stored in `checkpoints` folder with the final checkpoint (checkpoint_49999680.pt). Then start evaluating and store the video in `video` folder.

## Results

![Alt text](github_asset/boxing.gif?raw=true "Boxing Result") ![Alt text](github_asset/assault.gif?raw=true "Assault Result")

This work try to solve the spatial problem in Explainable Deep Reinforcement Learning. Our model is evaluated on ATARI benchmark. We found that our attention mask is accurate when overlayed on Visual Input and interpretable from all 57 ATARI games. The approach also integrated to [A3C LSTM](https://github.com/dgriff777/rl_a3c_pytorch) to prove the versality, please visit out [paper]() for detail of network architecture. For the full understanding of the interpretability evaluation, please visit our demo at [Interpretable Feature Extractor](https://sites.google.com/view/pay-attention-to-windows).

### Acknowledgements

Here are some other implementations and resources that were helpful in the completion of this project:

- https://github.com/schmidtdominik/Rainbow
- OpenAI Baselines (especially for preprocessing and Atari wrappers)
- https://github.com/dgriff777/rl_a3c_pytorch
- https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py
- https://github.com/Kaixhin/Rainbow/
- https://github.com/Kaixhin/Rainbow/wiki/Matteo's-Notes

If you use this code for any of your work, please cite our paper and give the proper credits for those works.

## Contributing

In the context of this project, we do not expect pull requests. If you find a bug, or would like to suggest an improvement, please open an issue.

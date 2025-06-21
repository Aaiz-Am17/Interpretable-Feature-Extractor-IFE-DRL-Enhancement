"""
This file defines all the neural network architectures available to use.
"""
from functools import partial
from math import sqrt
from ale_py import ALEInterface
#ale = ALEInterface()
import torch
from torch import nn as nn, Tensor
from torch.nn import init
import torch.nn.functional as F

class FactorizedNoisyLinear(nn.Module):
    """ The factorized Gaussian noise layer for noisy-nets dqn. """
    def __init__(self, in_features: int, out_features: int, sigma_0: float) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        # weight: w = \mu^w + \sigma^w . \epsilon^w
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        # bias: b = \mu^b + \sigma^b . \epsilon^b
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # initialization is similar to Kaiming uniform (He. initialization) with fan_mode=fan_in
        scale = 1 / sqrt(self.in_features)

        init.uniform_(self.weight_mu, -scale, scale)
        init.uniform_(self.bias_mu, -scale, scale)

        init.constant_(self.weight_sigma, self.sigma_0 * scale)
        init.constant_(self.bias_sigma, self.sigma_0 * scale)

    @torch.no_grad()
    def _get_noise(self, size: int) -> Tensor:
        noise = torch.randn(size, device=self.weight_mu.device)
        # f(x) = sgn(x)sqrt(|x|)
        return noise.sign().mul_(noise.abs().sqrt_())

    @torch.no_grad()
    def reset_noise(self) -> None:
        # like in eq 10 and 11 of the paper
        epsilon_in = self._get_noise(self.in_features)
        epsilon_out = self._get_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @torch.no_grad()
    def disable_noise(self) -> None:
        self.weight_epsilon[:] = 0
        self.bias_epsilon[:] = 0

    def forward(self, input: Tensor) -> Tensor:
        # y = wx + d, where
        # w = \mu^w + \sigma^w * \epsilon^w
        # b = \mu^b + \sigma^b * \epsilon^b
        return F.linear(input,
                        self.weight_mu + self.weight_sigma*self.weight_epsilon,
                        self.bias_mu + self.bias_sigma*self.bias_epsilon)

class Dueling(nn.Module):
    """ The dueling branch used in all nets that use dueling-dqn. """
    def __init__(self, value_branch, advantage_branch):
        super().__init__()
        self.flatten = nn.Flatten()
        self.value_branch = value_branch
        self.advantage_branch = advantage_branch

    def forward(self, x, advantages_only=False):
        x = self.flatten(x)
        advantages = self.advantage_branch(x)
        if advantages_only:
            return advantages

        value = self.value_branch(x)
        return value + (advantages - torch.mean(advantages, dim=1, keepdim=True))


class DuelingAlt(nn.Module):
    """ The dueling branch used in all nets that use dueling-dqn. """
    def __init__(self, l1, l2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            l1,
            nn.ReLU(),
            l2
        )

    def forward(self, x, advantages_only=False):
        res = self.main(x)
        advantages = res[:, 1:]
        value = res[:, 0:1]
        return value + (advantages - torch.mean(advantages, dim=1, keepdim=True))

class NatureCNN(nn.Module):
    """
    This is the CNN that was introduced in Mnih et al. (2013) and then used in a lot of later work such as
    Mnih et al. (2015) and the Rainbow paper. This implementation only works with a frame resolution of 84x84.
    """
    def __init__(self, depth, actions, linear_layer):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            linear_layer(3136, 512),
            nn.ReLU(),
            linear_layer(512, actions),
        )

    def forward(self, x, advantages_only=None):
        return self.main(x)


class DuelingNatureCNN(nn.Module):
    """
    Implementation of the dueling architecture introduced in Wang et al. (2015).
    This implementation only works with a frame resolution of 84x84.
    """
    def __init__(self, depth, actions, linear_layer):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.dueling = Dueling(
                nn.Sequential(linear_layer(3136, 512),
                              nn.ReLU(),
                              linear_layer(512, 1)),
                nn.Sequential(linear_layer(3136, 512),
                              nn.ReLU(),
                              linear_layer(512, actions))
            )

    def forward(self, x, advantages_only=False):
        f = self.main(x)
        return self.dueling(f, advantages_only=advantages_only)


class ImpalaCNNSmall(nn.Module):
    """
    Implementation of the small variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, depth, actions, linear_layer):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=depth, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        self.pool = torch.nn.AdaptiveMaxPool2d((6, 6))

        self.dueling = Dueling(
                nn.Sequential(linear_layer(1152, 256),
                              nn.ReLU(),
                              linear_layer(256, 1)),
                nn.Sequential(linear_layer(1152, 256),
                              nn.ReLU(),
                              linear_layer(256, actions))
            )

    def forward(self, x, advantages_only=False):
        f = self.main(x)
        f = self.pool(f)
        return self.dueling(f, advantages_only=advantages_only)


class ImpalaCNNResidual(nn.Module):
    """
    Simple residual block used in the large IMPALA CNN.
    """
    def __init__(self, depth, norm_func):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv_0 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))
        self.conv_1 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x_ = self.conv_0(self.relu(x))
        x_ = self.conv_1(self.relu(x_))
        return x+x_

class ImpalaCNNBlock(nn.Module):
    """
    Three of these blocks are used in the large IMPALA CNN.
    """
    def __init__(self, depth_in, depth_out, norm_func):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=depth_in, out_channels=depth_out, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)
        self.residual_0 = ImpalaCNNResidual(depth_out, norm_func=norm_func)
        self.residual_1 = ImpalaCNNResidual(depth_out, norm_func=norm_func)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.residual_0(x)
        x = self.residual_1(x)
        return x

# New implementation of HumanUnderstandableEncoding using 1x1 attention
class HumanUnderstandableEncoding(nn.Module):
    def __init__(self, num_inputs, channels_step):
        super(HumanUnderstandableEncoding, self).__init__()
        # Replace non-overlapping convolutions with 1x1 convolutions
        self.conv1x1_1 = nn.Conv2d(num_inputs, channels_step, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(channels_step, 2*channels_step, kernel_size=1)
        
        # Attention mechanism using 1x1 convolutions
        self.att_conv1 = nn.Conv2d(2*channels_step, channels_step, kernel_size=1)
        self.att_conv2 = nn.Conv2d(channels_step, 1, kernel_size=1)
        
        # Initialize weights
        relu_gain = nn.init.calculate_gain("relu")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.mul_(relu_gain)
                m.bias.data.fill_(0)
        
        self.att = []

    def forward(self, x):
        # Extract features with 1x1 convolutions
        features = F.relu(self.conv1x1_1(x))
        features = F.relu(self.conv1x1_2(features))
        
        # Compute attention weights
        attention = F.relu(self.att_conv1(features))
        attention = torch.sigmoid(self.att_conv2(attention))
        
        # Apply attention to features
        attended_features = features * attention
        
        # Store attention for visualization
        self.att = attention.squeeze(1)
        
        return attended_features

# Depthwise Separable Convolution for MobileNet
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                  stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# MobileNet-based AFE to replace IMPALA CNN in ImpalaCNNLarge
class ImpalaCNNLarge(nn.Module):
    """
    Implementation of the Interpretable Feature Extractor with modified architecture:
    - HUE with 1x1 attention mechanisms
    - AFE with MobileNet instead of IMPALA CNN
    """
    def __init__(self, in_depth, actions, linear_layer, model_size=1, spectral_norm=False):
        super().__init__()

        # Human Understandable Encoding with 1x1 attention
        self.HUE = HumanUnderstandableEncoding(in_depth, 16*model_size)
        
        # Agent-Friendly Encoding using MobileNet architecture
        self.AFE = nn.Sequential(
            nn.Conv2d(32*model_size, 32*model_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32*model_size),
            nn.ReLU(),
            
            DepthwiseSeparableConv(32*model_size, 64*model_size),
            nn.BatchNorm2d(64*model_size),
            nn.ReLU(),
            
            DepthwiseSeparableConv(64*model_size, 128*model_size, stride=2),
            nn.BatchNorm2d(128*model_size),
            nn.ReLU(),
            
            DepthwiseSeparableConv(128*model_size, 128*model_size),
            nn.BatchNorm2d(128*model_size),
            nn.ReLU(),
            
            DepthwiseSeparableConv(128*model_size, 256*model_size, stride=2),
            nn.BatchNorm2d(256*model_size),
            nn.ReLU(),
            
            nn.AdaptiveMaxPool2d((8, 8))
        )
        
        # Keep the dueling architecture for compatibility
        self.dueling = Dueling(
            nn.Sequential(linear_layer(2048*model_size, 256),
                          nn.ReLU(),
                          linear_layer(256, 1)),
            nn.Sequential(linear_layer(2048*model_size, 256),
                          nn.ReLU(),
                          linear_layer(256, actions))
        )

    def forward(self, x, advantages_only=False):
        f = self.HUE(x)
        f = self.AFE(f)
        return self.dueling(f, advantages_only=advantages_only)

# Keep the original SoftAttention class for backward compatibility
class SolfAttention(nn.Module):
    """
    Attention Network.
    """
    def __init__(self, features_dim, attention_dim):
        """
        :param features_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(SolfAttention, self).__init__()
        self.encoder_att = nn.Linear(features_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features):
        """
        Forward propagation.
        :param features: encoded images, a tensor of dimension (batch_size, num_pixels, features_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(features)
        att = self.full_att(self.relu(att1)).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (features * alpha.unsqueeze(2))

        return attention_weighted_encoding, alpha

def get_model(model_str, spectral_norm):
    if model_str == 'nature': return NatureCNN
    elif model_str == 'dueling': return DuelingNatureCNN
    elif model_str == 'impala_small': return ImpalaCNNSmall
    elif model_str.startswith('impala_large:'):
        return partial(ImpalaCNNLarge, model_size=int(model_str[13:]), spectral_norm=spectral_norm)
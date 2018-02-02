import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

# start to define the network...
# actor critic style, but without entropy regularization
class Actor_Network(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Actor_Network, self).__init__()
        self.affine_1 = nn.Linear(num_inputs, 400)
        self.affine_2 = nn.Linear(400, 300)

        self.action_alpha = nn.Linear(300, num_actions)
        self.action_beta = nn.Linear(300, num_actions)

        # init the networks....
        self.action_alpha.weight.data.mul_(0.1)
        self.action_alpha.bias.data.mul_(0.0)

        self.action_beta.weight.data.mul_(0.1)
        self.action_alpha.bias.data.mul_(0.0)


    def forward(self, x):
        x = F.relu(self.affine_1(x))
        x = F.relu(self.affine_2(x))
            
        action_alpha = F.softplus(self.action_alpha(x)) + 1
        action_beta = F.softplus(self.action_beta(x)) + 1

        return action_alpha, action_beta

# define the critic network....
class Critic_Network(nn.Module):
    def __init__(self, num_inputs):
        super(Critic_Network, self).__init__()
        self.affine_1 = nn.Linear(num_inputs, 400)
        self.affine_2 = nn.Linear(400, 300)

        self.value_head = nn.Linear(300, 1)
        
        # init the network...
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.affine_1(x))
        x = F.relu(self.affine_2(x))

        value = self.value_head(x)

        return value


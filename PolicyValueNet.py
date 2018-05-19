# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch (tested in PyTorch 0.2.0 and 0.3.0)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class ConvBlock(nn.Module):
    '''Convolutional Block'''
    def __init__(self, in_channels=4, out_channels=256):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ResidualBlock(nn.Module):
    '''Residual Block'''
    def __init__(self, out_channels=128): # input_channels=output_channels
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += x # skip connection that adds the input to the block
        out = self.relu2(out)
        return out



class ResNet(nn.Module):
    '''One Block ResNet According to the paper'''
    def __init__(self, board_width, board_height, in_channels=4, out_channels=128):
        super(ResNet, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv_layer = ConvBlock(in_channels, out_channels)
        self.res_layer = self.make_residual_layers(1, out_channels) # 论文里blocks=19 or 39

        # policy head: action policy layers
        self.act_filters = 2
        self.act_conv1 = nn.Conv2d(out_channels, self.act_filters, kernel_size=1, stride=1) #2 filters
        self.act_bn1 = nn.BatchNorm2d(self.act_filters)
        self.act_relu1 = nn.LeakyReLU()
        self.act_fc1 = nn.Linear(self.act_filters * board_width * board_height, board_width * board_height)
        self.act_softmax = nn.Softmax(dim=1)

        # value head: state value layers
        self.val_filters = 1
        self.val_hidden_num = 128
        self.val_conv1 = nn.Conv2d(out_channels, self.val_filters , kernel_size=1)
        self.val_bn1 = nn.BatchNorm2d(self.val_filters)
        self.val_relu1 = nn.LeakyReLU()
        self.val_fc1 = nn.Linear(self.val_filters * board_width * board_height, self.val_hidden_num)
        self.val_relu2 = nn.LeakyReLU()
        self.val_fc2 = nn.Linear(self.val_hidden_num, 1)
        self.val_tanh = nn.Tanh()


    def make_residual_layers(self, blocks=2, out_channels=256):
        layers = []
        for i in range(blocks):
            layers.append(ResidualBlock(out_channels))
        return nn.Sequential(*layers)


    def forward(self, state):
        # common layer
        x = self.conv_layer(state)
        x = self.res_layer(x)
        # policy head
        x_act = self.act_conv1(x)
        x_act = self.act_bn1(x_act)
        x_act = self.act_relu1(x_act)
        x_act = x_act.view(-1, self.act_filters * self.board_width * self.board_height)#flatten
        policy_logits = self.act_fc1(x_act)
        policy_output = self.act_softmax(policy_logits)

        # value head
        x_val = self.val_conv1(x)
        x_val = self.val_bn1(x_val)
        x_val = self.val_relu1(x_val)
        x_val = x_val.view(-1, self.val_filters * self.board_width * self.board_height)
        x_val = self.val_fc1(x_val)
        x_val = self.val_relu2(x_val)
        x_val = self.val_fc2(x_val)
        value_output = self.val_tanh(x_val)
        return policy_logits, policy_output, value_output


    def __str__(self):
        return "resnet"


class ConvNet(nn.Module):
    """Conv Layers"""

    def __init__(self, board_width, board_height):
        super(ConvNet, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        n = 1 # 最开始是1
        common_kernel_size = 2 * n + 1
        self.conv1 = nn.Conv2d(4, 32, kernel_size=common_kernel_size, padding=n)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=common_kernel_size, padding=n)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=common_kernel_size, padding=n)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        policy_logits = self.act_fc1(x_act)
        policy_output = F.softmax(policy_logits, dim=1) # log是因为用库可以避免输出概率为0等特殊情况。否则如果后续用torch.log处理，会出现loss=nan的情况

        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        value_output = F.tanh(self.val_fc2(x_val))

        return policy_logits, policy_output, value_output

    def __str__(self):
        return "conv_net"


class FeedForwardNet(nn.Module):
    '''Feed Forward Network'''
    def __init__(self, board_width, board_height):
        super(FeedForwardNet, self).__init__()
        self.board_width = board_width
        self.board_height = board_height

        # common layers
        self.fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)

        # action policy layers
        self.act_fc1 = nn.Linear(board_width * board_height, board_width * board_height)

        # state value layers
        self.val_fc1 = nn.Linear(board_width * board_height, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.fc1(state_input.view(-1, 4 * self.board_width * self.board_height)))

        # action policy layers
        policy_logits = F.relu(self.act_fc1(x))
        policy_output = F.softmax(policy_logits, dim=1)  # log是因为用库可以避免输出概率为0等特殊情况。否则如果后续用torch.log处理，会出现loss=nan的情况

        # state value layers
        value_output = F.tanh(self.val_fc1(x))

        return policy_logits, policy_output, value_output


class ResNet2(nn.Module):
    '''Two Block ResNet According to the paper'''
    def __init__(self, board_width, board_height, in_channels=4, out_channels=128):
        super(ResNet2, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv_layer = ConvBlock(in_channels, out_channels)
        self.res_layer = self.make_residual_layers(2, out_channels) # 论文里blocks=19 or 39

        # policy head: action policy layers
        self.act_filters = 2
        self.act_conv1 = nn.Conv2d(out_channels, self.act_filters, kernel_size=1, stride=1) #2 filters
        self.act_bn1 = nn.BatchNorm2d(self.act_filters)
        self.act_relu1 = nn.ReLU()
        self.act_fc1 = nn.Linear(self.act_filters * board_width * board_height, board_width * board_height)
        self.act_softmax = nn.Softmax(dim=1)

        # value head: state value layers
        self.val_filters = 2
        self.val_hidden_num = 256
        self.val_conv1 = nn.Conv2d(out_channels, self.val_filters , kernel_size=1)
        self.val_bn1 = nn.BatchNorm2d(self.val_filters)
        self.val_relu1 = nn.ReLU()
        self.val_fc1 = nn.Linear(self.val_filters * board_width * board_height, self.val_hidden_num)
        self.val_relu2 = nn.ReLU()
        self.val_fc2 = nn.Linear(self.val_hidden_num, 1)
        self.val_tanh = nn.Tanh()


    def make_residual_layers(self, blocks=2, out_channels=256):
        layers = []
        for i in range(blocks):
            layers.append(ResidualBlock(out_channels))
        return nn.Sequential(*layers)


    def forward(self, state):
        # common layer
        x = self.conv_layer(state)
        x = self.res_layer(x)
        # policy head
        x_act = self.act_conv1(x)
        x_act = self.act_bn1(x_act)
        x_act = self.act_relu1(x_act)
        x_act = x_act.view(-1, self.act_filters * self.board_width * self.board_height)#flatten
        policy_logits = self.act_fc1(x_act)
        policy_output = self.act_softmax(policy_logits)

        # value head
        x_val = self.val_conv1(x)
        x_val = self.val_bn1(x_val)
        x_val = self.val_relu1(x_val)
        x_val = x_val.view(-1, self.val_filters * self.board_width * self.board_height)
        x_val = self.val_fc1(x_val)
        x_val = self.val_relu2(x_val)
        x_val = self.val_fc2(x_val)
        value_output = self.val_tanh(x_val)
        return policy_logits, policy_output, value_output


    def __str__(self):
        return "resnet"


"""policy-value network wrapper """
class PolicyValueNet():

    def __init__(self, board_width, board_height, net_params=None, Network=None, use_gpu=False):
        if Network is None: Network = ResNet
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty 
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Network(board_width, board_height).cuda()
        else:
            self.policy_value_net = Network(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        if net_params:
            self.policy_value_net.load_state_dict(net_params)


    def predict_many(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            _, policy_output, value_output = self.policy_value_net(state_batch)
            return policy_output.data.cpu().numpy(), value_output.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            _, policy_output, value_output = self.policy_value_net(state_batch)
            return policy_output.data.numpy(), value_output.data.numpy()


    def predict(self, board):
        """
        input: board：a single sample
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        legal_positions = board.availables
        # (batch, channels, width, height)
        current_state = np.array(board.current_state().reshape(-1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            _, policy_output, value_output = self.policy_value_net(Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = policy_output.data.cpu().numpy().flatten()
        else:
            # probs:(batch_size, width*height); value:(batch_size, 1)
            _, policy_output, value_output = self.policy_value_net(Variable(torch.from_numpy(current_state)).float())
            act_probs = policy_output.data.numpy().flatten()

        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value_output.data[0][0]

    def fit(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        policy_logits,_, value_output = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2 (Note: the L2 penalty is incorporated in optimizer)
        value_loss = F.mse_loss(value_output.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * F.log_softmax(policy_logits,dim=1), 1))

        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only，- sum (p*logp)
        log_policy_output = F.log_softmax(policy_logits, dim=1) # 为了防止手动处理log时p负数问题，故先调用库函数
        entropy = -torch.mean(torch.sum(torch.exp(log_policy_output) * log_policy_output, 1))

        # entropy is equivalent to policy loss.
        return {
            'combined_loss':loss.data[0],
            'policy_loss': policy_loss.data[0],
            'value_loss':value_loss.data[0],
            'entropy': entropy.data[0]
        }



    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def __str__(self):
        return self.policy_value_net.__str__()

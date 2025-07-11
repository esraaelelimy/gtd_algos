## This code is copied from https://github.com/mohmdelsayed/streaming-drl/blob/main/stream_q_minatar.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ObGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        super(ObGD, self).__init__(params, defaults)

    def step(self, delta, reset=False):
        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)
                z_sum += e.abs().sum().item()

        delta_bar = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]
        if dot_product > 1:
            step_size = group["lr"] / dot_product
        else:
            step_size = group["lr"]

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                p.data.add_(delta * e, alpha=-step_size)
                if reset:
                    e.zero_()


def sparse_init(tensor, sparsity, type='uniform'):
    if tensor.ndimension() == 2:
        fan_out, fan_in = tensor.shape

        num_zeros = int(math.ceil(sparsity * fan_in))

        with torch.no_grad():
            if type == 'uniform':
                tensor.uniform_(-math.sqrt(1.0 / fan_in), math.sqrt(1.0 / fan_in))
            elif type == 'normal':
                tensor.normal_(0, math.sqrt(1.0 / fan_in))
            else:
                raise ValueError("Unknown initialization type")
            for col_idx in range(fan_out):
                row_indices = torch.randperm(fan_in)
                zero_indices = row_indices[:num_zeros]
                tensor[col_idx, zero_indices] = 0
        return tensor

    elif tensor.ndimension() == 4:
        channels_out, channels_in, h, w = tensor.shape
        fan_in, fan_out = channels_in*h*w, channels_out*h*w

        num_zeros = int(math.ceil(sparsity * fan_in))

        with torch.no_grad():
            if type == 'uniform':
                tensor.uniform_(-math.sqrt(1.0 / fan_in), math.sqrt(1.0 / fan_in))
            elif type == 'normal':
                tensor.normal_(0, math.sqrt(1.0 / fan_in))
            else:
                raise ValueError("Unknown initialization type")
            for out_channel_idx in range(channels_out):
                indices = torch.randperm(fan_in)
                zero_indices = indices[:num_zeros]
                tensor[out_channel_idx].reshape(channels_in*h*w)[zero_indices] = 0
        return tensor

    else:
        raise ValueError("Only tensors with 2 or 4 dimensions are supported")


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class LayerNormalization(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return F.layer_norm(input, input.size())
    def extra_repr(self) -> str:
        return "Layer Normalization"


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)


class StreamQ(nn.Module):
    def __init__(self, n_channels=4, n_actions=3, hidden_size=128, lr=1.0, epsilon_target=0.01, epsilon_start=1.0, exploration_fraction=0.1, total_steps=1_000_000, gamma=0.99, lamda=0.8, kappa_value=2.0):
        super(StreamQ, self).__init__()
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_target = epsilon_target
        self.epsilon = epsilon_start
        self.exploration_fraction = exploration_fraction
        self.total_steps = total_steps
        self.time_step = 0
        self.network = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, stride=1),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=0),
            nn.Linear(1024, hidden_size),
            LayerNormalization(),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, n_actions)
        )
        self.apply(initialize_weights)
        self.optimizer = ObGD(list(self.parameters()), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

    def q(self, x):
        x = torch.moveaxis(x, -1, 0)
        return self.network(x)

    def sample_action(self, s):
        self.time_step += 1
        self.epsilon = linear_schedule(self.epsilon_start, self.epsilon_target, self.exploration_fraction * self.total_steps, self.time_step)
        if isinstance(s, np.ndarray):
            s = torch.tensor(np.array(s), dtype=torch.float)
        if np.random.rand() < self.epsilon:
            q_values = self.q(s)
            greedy_action = torch.argmax(q_values, dim=-1).item()
            random_action = np.random.randint(0, self.n_actions)
            if greedy_action == random_action:
                return random_action, False
            else:
                return random_action, True
        else:
            q_values = self.q(s)
            return torch.argmax(q_values, dim=-1), False

    def update_params(self, s, a, r, s_prime, done, is_nongreedy, overshooting_info=False):
        done_mask = 0 if done else 1
        s, a, r, s_prime, done_mask = torch.tensor(np.array(s), dtype=torch.float), torch.tensor([a], dtype=torch.int).squeeze(0), \
                                         torch.tensor(np.array(r)), torch.tensor(np.array(s_prime), dtype=torch.float), \
                                         torch.tensor(np.array(done_mask), dtype=torch.float)
        
        q_sa = self.q(s)[a]
        max_q_s_prime_a_prime = torch.max(self.q(s_prime), dim=-1).values
        td_target = r + self.gamma * max_q_s_prime_a_prime * done_mask
        delta = td_target - q_sa

        q_output = -q_sa
        self.optimizer.zero_grad()
        q_output.backward()
        self.optimizer.step(delta.item(), reset=(done or is_nongreedy))

        if overshooting_info:
            max_q_s_prime_a_prime = torch.max(self.q(s_prime), dim=-1).values
            td_target = r + self.gamma * max_q_s_prime_a_prime * done_mask
            delta_bar = td_target - self.q(s)[a]
            if torch.sign(delta_bar * delta).item() == -1:
                print("Overshooting Detected!")

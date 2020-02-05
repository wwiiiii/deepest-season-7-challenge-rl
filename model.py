import torch
import torch.nn as nn
import torch.nn.functional as F


class CartPoleNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CartPolePolicy:
    def __init__(self, input_size, output_size, device='cpu'):
        self.net = CartPoleNet(input_size, output_size).to(device)
        if torch.cuda.device_count() >= 2 and device == 'cuda':
            self.net = nn.DataParallel(self.net)

    def __call__(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def copy_params_(self, other):
        assert isinstance(other, CartPolePolicy)
        self.net.load_state_dict(other.net.state_dict())

    def parameters(self):
        return self.net.parameters()

    def get_greedy(self, state):
        q = self.net(state.float())
        _, a = torch.max(q, 1)
        return a

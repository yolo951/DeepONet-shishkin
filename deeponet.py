import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

#1d setting
class DeepONet(nn.Module):
    def __init__(self, b_dim, t_dim):
        super(DeepONet, self).__init__()
        self.b_dim = b_dim
        self.t_dim = t_dim

        self.branch = nn.Sequential(
            nn.Linear(self.b_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 128),
        )

        self.trunk = nn.Sequential(
            nn.Linear(self.t_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 128),
        )

        self.b = Parameter(torch.zeros(1))

    def forward(self, x, l):
        x = self.branch(x)
        l = self.trunk(l)

        res = torch.einsum("bi,bi->b", x, l)
        res = res.unsqueeze(1) + self.b
        return res

#2d setting

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

class DeepONet2D(nn.Module):
    def __init__(self,b_dim,t_dim):
        super(DeepONet2D, self).__init__()
        self.b_dim = b_dim
        self.t_dim = t_dim

        self.branch = nn.Sequential(
            nn.Linear(self.b_dim, 65*65),
            Reshape(1,65,65),
            nn.Conv2d(1, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2),
            nn.ReLU(),
            Reshape(128*14*14),
            nn.Linear(128*14*14, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
            )
        # self.branch = nn.Sequential(
        #     nn.Linear(self.b_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        # )
        self.trunk = nn.Sequential(
            nn.Linear(self.t_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )

        self.b = Parameter(torch.zeros(1))


    def forward(self, x, l):
        x = self.branch(x)
        l = self.trunk(l)

        res = torch.einsum("bi,bki->bk", x, l)
        res = res.unsqueeze(-1) + self.b
        return res

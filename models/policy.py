import torch.nn as nn
import torch.nn.functional as F

from models.base import ModelBase

class DiscretePolicy(ModelBase):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super().__init__(num_inputs, num_outputs)
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        x = x.view(1, -1)
        x = F.relu(self.linear1(x))
        action_scores = self.linear2(x)
        return F.softmax(action_scores, dim=1)


class ContinuousPolicy(ModelBase):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super().__init__(num_inputs, num_outputs)
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)
        self.linear2_ = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma_sq = self.linear2_(x)

        return mu, sigma_sq

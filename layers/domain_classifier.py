import torch.nn as nn


class DomainClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, output_size),
        )

    def forward(self, h):
        y = self.layer(h)
        return y

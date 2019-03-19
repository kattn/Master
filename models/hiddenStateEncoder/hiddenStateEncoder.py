import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tools

torch.manual_seed(1)

bidirectional = True
numLSTMLayers = 2
dropout = 0.1

outputFunction = nn.Tanh()

lr = 0.01
lossFunction = nn.MSELoss()
optimizer = optim.Adam

numEpochs = 1000


class HiddenStateEncoder(nn.Module):
    """
    Uses a linear layer to map the sensor features into a hidden state. This
    is a seperate module as it might be benefictial to train by itself.
    """
    def __init__(self):
        super(HiddenStateEncoder, self).__init__(inputSize, hiddenStateSize)
        self.output = outputFunction
        self.encoder = nn.Linear(inputSize, hiddenStateSize)

        self.path = "models/hiddenStateEncoder/hiddenStateEncoder.pt"

    def forward(self, inp):
        output = self.encoder(inp)
        output = self.output(output)
        return output

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import settings
import tools

# Set seed for reproducability for my code, will not transfer

# Input size is set to the number of pressure sensors
# The hidden size is tested with 5, 10, 15 and 20
# Bidirectional was not used
# Dropout was not used
# Number of layers was tested with 1, 2, 3
inputSize = tools.getNumSensors("p")
hiddenSize = 50
bidirectional = False
numLayers = 3
dropout = 0

outputFunction = nn.Softmax(dim=settings.numClasses)


class SingleGRU(nn.Module):
    lr = 0.03
    targetWeight = torch.tensor([0.325, 0.675])
    lossFunction = nn.CrossEntropyLoss(weight=targetWeight)
    optimizer = optim.Adam

    numEpochs = 25

    def __init__(self):
        super(SingleGRU, self).__init__()
        self.hidden = self.init_hidden()
        self.output = outputFunction
        self.gru = nn.GRU(
            input_size=inputSize, hidden_size=hiddenSize,
            num_layers=numLayers, bidirectional=bidirectional,
            dropout=dropout)
        self.decoder = nn.Linear(
            hiddenSize*(bidirectional+1), settings.numClasses)

        self.modelPath = "gru_hs" + str(hiddenSize) + "_nL" + str(numLayers) + ".pt"

    def init_hidden(self, hidden=None):
        if hidden is not None:
            self.hidden = hidden
        else:
            self.hidden = (
                torch.randn(
                    numLayers*(bidirectional+1), 1, hiddenSize))
        return self.hidden

    def forward(self, inp):
        if isinstance(inp, tuple):
            inp = inp[0]

        output, self.hidden = self.gru(inp, self.hidden.detach())
        output = self.decoder(output)
        # output = self.output(output)
        return output

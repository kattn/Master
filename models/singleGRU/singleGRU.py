import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tools

torch.manual_seed(1)

inputSize = tools.getNumSensors("t")
hiddenSize = 10
bidirectional = True
numLayers = 2
dropout = 0

outputFunction = nn.Sigmoid()


class SingleGRU(nn.Module):
    lr = 0.01
    lossFunction = nn.BCELoss()
    optimizer = optim.Adam

    numEpochs = 10

    def __init__(self):
        super(SingleGRU, self).__init__()
        self.hidden = self.init_hidden()
        self.output = outputFunction
        self.gru = nn.GRU(
            input_size=inputSize, hidden_size=hiddenSize,
            num_layers=numLayers, bidirectional=bidirectional,
            dropout=dropout)
        self.decoder = nn.Linear(
            hiddenSize*(bidirectional+1), tools.numClasses)

        self.modelPath = __file__.replace(os.getcwd(), "")[1:-3] + ".pt"

    def init_hidden(self, hidden=None):
        if hidden is not None:
            self.hidden = hidden
        else:
            self.hidden = (
                torch.randn(
                    numLayers*(bidirectional+1), 1, hiddenSize))

        return self.hidden

    def forward(self, inp):
        output, self.hidden = self.gru(inp, self.hidden.detach())
        output = self.decoder(output)
        output = self.output(output)
        return output

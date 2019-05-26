import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import settings
import tools

# Set seed for reproducability for my code, will not transfer
torch.manual_seed(1)

# Input size is set to the number of pressure sensors
# The hidden size is tested with 5, 10, 15 and 20
# Bidirectional was not used
# Dropout was not used
# Number of layers was tested with 1, 2, 3
inputSize = tools.getNumSensors("p")
hiddenSize = 20
bidirectional = False
numLayers = 3
dropout = 0

outputFunction = nn.Sigmoid()


class SingleLSTM(nn.Module):
    lr = 0.01
    lossFunction = nn.BCELoss()
    optimizer = optim.Adam

    numEpochs = 50

    def __init__(self):
        super(SingleLSTM, self).__init__()
        self.hidden = self.init_hidden()
        self.output = outputFunction
        self.lstm = nn.LSTM(
            input_size=inputSize, hidden_size=hiddenSize,
            num_layers=numLayers, bidirectional=bidirectional,
            dropout=dropout)
        self.decoder = nn.Linear(
            hiddenSize*(bidirectional+1), settings.numClasses)

        self.modelPath = "lstm_hs" + str(hiddenSize) + "_nL" + str(numLayers) + ".pt"

    def init_hidden(self, hidden=None):
        if hidden is not None:
            self.hidden = hidden
        else:
            self.hidden = (
                torch.randn(
                    numLayers*(bidirectional+1), 1, hiddenSize),
                torch.randn(
                    numLayers*(bidirectional+1), 1, hiddenSize))

        return self.hidden

    def forward(self, inp):
        if isinstance(inp, tuple):
            inp = inp[0]

        output, self.hidden = self.lstm(inp, (self.hidden[0].detach(), self.hidden[1].detach()))
        output = self.decoder(output)
        output = self.output(output)
        return output

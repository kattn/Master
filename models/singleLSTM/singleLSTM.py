import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tools

torch.manual_seed(1)

inputSize = tools.getNumSensors("t")
hiddenSize = 45
bidirectional = True
numLSTMLayers = 2
dropout = 0.1

outputFunction = nn.Sigmoid()

lr = 0.01
lossFunction = nn.MSELoss()
optimizer = optim.Adam

numEpochs = 1000


class SingleLSTM(nn.Module):
    def __init__(self):
        super(SingleLSTM, self).__init__()
        self.hidden = self.init_hidden()
        self.output = outputFunction
        self.lstm = nn.LSTM(
            input_size=inputSize, hidden_size=hiddenSize,
            num_layers=numLSTMLayers, bidirectional=bidirectional,
            dropout=dropout)
        self.decoder = nn.Linear(
            hiddenSize*(bidirectional+1), tools.numClasses)

        self.path = "models/singleLSTM.pt"

    def init_hidden(self, hidden=None):
        if hidden:
            self.hidden = hidden
        else:
            self.hidden = (
                torch.zeros(
                    numLSTMLayers*(bidirectional+1), 1, hiddenSize),
                torch.zeros(
                    numLSTMLayers*(bidirectional+1), 1, hiddenSize))

        return self.hidden

    def forward(self, inp):
        output, self.hidden = self.lstm(inp, self.hidden)
        output = self.decoder(output)
        output = self.output(output)
        return output, self.hidden

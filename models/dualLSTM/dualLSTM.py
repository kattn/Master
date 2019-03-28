import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tools

torch.manual_seed(1)

lInputSize = tools.getNumSensors("p")
lHiddenSize = 45
rInputSize = tools.getNumSensors("f")
rHiddenSize = 45

bidirectional = True
numLSTMLayers = 2
dropout = 0.1

outputFunction = nn.Sigmoid()

lr = 0.01
lossFunction = nn.MSELoss()
optimizer = optim.Adam

numEpochs = 1000


class DualLSTM(nn.Module):
    def __init__(self):
        super(DualLSTM, self).__init__()
        self.lHidden, self.rHidden = self.init_hidden()
        self.output = outputFunction
        self.lLstm = nn.LSTM(
            input_size=lInputSize, hidden_size=lHiddenSize,
            num_layers=numLSTMLayers, bidirectional=bidirectional,
            dropout=dropout)
        self.rLstm = nn.LSTM(
            input_size=rInputSize, hidden_size=rHiddenSize,
            num_layers=numLSTMLayers, bidirectional=bidirectional,
            dropout=dropout)
        self.decoder = nn.Linear(
            lHiddenSize*(bidirectional+1)+rHiddenSize*(bidirectional+1),
            tools.numClasses)

        self.modelPath = __file__.replace(os.getcwd(), "")[1:-3] + ".pt"

    def init_hidden(self, hidden=None):
        if hidden:
            self.hidden = hidden
        else:
            self.lHidden = (
                torch.zeros(numLSTMLayers*(bidirectional+1), 1, lHiddenSize),
                torch.zeros(numLSTMLayers*(bidirectional+1), 1, lHiddenSize))
            self.rHidden = (
                torch.zeros(numLSTMLayers*(bidirectional+1), 1, rHiddenSize),
                torch.zeros(numLSTMLayers*(bidirectional+1), 1, rHiddenSize))

        return (self.lHidden, self.rHidden)

    def forward(self, inp):
        """
        Takes a touple of inputs and outputs the output and two touples, the
        output tuple and the hidden state tuples.
        """
        lInp, rInp = inp
        lOutput, self.lHidden = self.lLstm(lInp, self.lHidden)
        rOutput, self.rHidden = self.rLstm(rInp, self.rHidden)
        merged = torch.cat((lOutput, rOutput), 2)
        output = self.output(self.decoder(merged))
        return output, (self.lHidden, self.rHidden)

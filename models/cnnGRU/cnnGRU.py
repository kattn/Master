import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tools

torch.manual_seed(1)

kernelSize = 5
hiddenSize = 10
# since the GRU gets inp from two CNNs
inpSize = tools.getNumSensors("t") - (kernelSize-1)*2
bidirectional = True
numLayers = 2
dropout = 0.1

outputFunction = nn.Sigmoid()


class CNNGRU(nn.Module):
    lr = 0.006
    lossFunction = nn.BCELoss()
    optimizer = optim.Adam

    numEpochs = 10

    def __init__(self):
        super(CNNGRU, self).__init__()
        self.hidden = self.init_hidden()
        self.output = outputFunction
        self.presCNN = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=1,
                kernel_size=kernelSize),
            nn.ReLU()
        )
        self.flowCNN = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=1,
                kernel_size=kernelSize),
            nn.ReLU()
        )
        self.gru = nn.GRU(
            input_size=inpSize, hidden_size=hiddenSize,
            num_layers=numLayers, bidirectional=bidirectional)
        self.decoder = nn.Linear(
            hiddenSize*(bidirectional+1), tools.numClasses)

        self.path = "models/CNNGRU/cnngru.pt"

    def init_hidden(self, hidden=None):
        if hidden:
            self.hidden = hidden
        else:
            self.hidden = torch.zeros(
                    numLayers*(bidirectional+1), 1, hiddenSize)

        return self.hidden

    def forward(self, inp):
        presInp = inp[0]
        flowInp = inp[1]

        presOut = self.presCNN(presInp)
        flowOut = self.flowCNN(flowInp)

        gruInp = torch.cat((presOut, flowOut), 2)

        output, self.hidden = self.gru(gruInp, self.hidden)
        output = self.decoder(output)
        output = self.output(output)
        return output, self.hidden

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tools

torch.manual_seed(1)

presSens = tools.getNumSensors("p")
kernelSize = 24  # hours

gruInp = tools.getNumSensors("p") - (kernelSize-1)
hiddenSize = 20
bidirectional = False
numLayers = 2
dropout = 0.1

outputFunction = nn.Sigmoid()


class CNNGRU(nn.Module):
    lr = 0.003
    lossFunction = nn.BCELoss()
    optimizer = optim.Adam

    numEpochs = 10

    def __init__(self):
        super(CNNGRU, self).__init__()
        self.name = input("Name the model:")
        self.hidden = self.init_hidden()
        self.output = outputFunction
        self.presCNN = nn.Sequential(
            nn.Conv1d(
                in_channels=presSens, out_channels=gruInp,
                kernel_size=kernelSize),
            nn.ReLU()
        )
        self.gru = nn.GRU(
            input_size=inpSize, hidden_size=hiddenSize,
            num_layers=numLayers, bidirectional=bidirectional)
        self.decoder = nn.Linear(
            hiddenSize*(bidirectional+1), tools.numClasses)

        self.modelPath = __file__.replace(os.getcwd(), "")[1:-3] + ".pt"

    def init_hidden(self, hidden=None):
        if hidden:
            self.hidden = hidden
        else:
            self.hidden = torch.randn(
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

    def classify(self, inp):
        output = self(inp)

        if str(self.lossFunction.__class__()) == "BCELoss()":
            classification = output.ge(0.5)
        elif str(self.lossFunction.__class__()) == "MSELoss()":
            classification = output.ge(0.5)
        elif str(self.lossFunction.__class__()) == "L1Loss()":
            classification = output.ge(0.5)
        elif str(self.lossFunction.__class__()) == "CrossEntropyLoss()":
            classification = output.max(2)[1]
        else:
            raise Exception("No classification for " + str(self.lossFunction.__class__()))

        return output, classification

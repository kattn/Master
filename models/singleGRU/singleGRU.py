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
hiddenSize = 20
bidirectional = False
numLayers = 2
dropout = 0.3

# outputFunction = nn.Softmax(dim=settings.numClasses)
outputFunction = nn.Sigmoid()


class SingleGRU(nn.Module):
    lr = 0.003
    targetWeight = torch.tensor([0.36, 0.64])
    # lossFunction = nn.CrossEntropyLoss(weight=targetWeight)
    lossFunction = nn.BCELoss()
    optimizer = optim.Adam

    numEpochs = 50

    def __init__(self):
        super(SingleGRU, self).__init__()
        self.name = input("Name the model:")
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

        output, self.hidden = self.gru(inp, self.hidden)
        # print("output")
        # print(output)
        output = self.decoder(output)
        # print("decoderOutput")
        # print(output)
        output = self.output(output)
        # print("sigmoidOutput")
        # print(output)
        return output

    # Returns the original output and classification
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

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

import tools
import settings


torch.manual_seed(1)

kernelSize = 6
presChannels = tools.getNumSensors("p")
outChannels = 3
padding = 0
dilation = 1
stride = 1
convLin = settings.sequenceSize
convLout = math.floor((convLin + 2 * padding - dilation * (kernelSize-1) - 1)/stride + 1)
decoderInputSize = math.floor((convLout + 2 * padding - dilation * (kernelSize-1) - 1)/stride + 1)*outChannels
dropout = 0.1


outputFunction = nn.Sigmoid()


class CNN(nn.Module):
    lr = 0.003
    lossFunction = nn.BCELoss()
    optimizer = optim.Adam

    numEpochs = 1

    def __init__(self):
        super(CNN, self).__init__()
        self.output = outputFunction
        self.presCNN = nn.Sequential(
            nn.Conv1d(
                in_channels=presChannels, out_channels=outChannels,
                kernel_size=kernelSize, stride=stride, padding=padding,
                dilation=dilation),
            nn.ReLU(),
            nn.MaxPool1d(kernelSize, stride=stride),
        )
        self.decoder = nn.Linear(
            decoderInputSize,
            settings.numClasses)

        self.modelPath = __file__.replace(os.getcwd(), "")[1:-3] + ".pt"

    def forward(self, inp):
        presInp = inp[0].transpose(1, 2).transpose(0, 2)
        flowInp = inp[1].transpose(1, 2).transpose(0, 2)

        output = self.presCNN(presInp)
        output = output.view(-1, 1, decoderInputSize)

        output = self.decoder(output)
        output = self.output(output)

        return output

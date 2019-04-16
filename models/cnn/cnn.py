import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tools

torch.manual_seed(1)

kernelSize = 6
presChannels = tools.getNumSensors("p")
flowChannels = tools.getNumSensors("f")
outChannels = 1
dropout = 0.1

outputFunction = nn.Sigmoid()


class CNN(nn.Module):
    lr = 0.003
    lossFunction = nn.BCELoss()
    optimizer = optim.Adam

    numEpochs = 20

    def __init__(self):
        super(CNN, self).__init__()
        self.output = outputFunction
        self.presCNN = nn.Sequential(
            nn.Conv1d(
                in_channels=presChannels, out_channels=presChannels,
                kernel_size=kernelSize),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=presChannels, out_channels=outChannels,
                kernel_size=kernelSize),
            nn.ReLU()
        )
        self.flowCNN = nn.Sequential(
            nn.Conv1d(
                in_channels=flowChannels, out_channels=flowChannels,
                kernel_size=kernelSize),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=flowChannels, out_channels=outChannels,
                kernel_size=kernelSize),
            nn.ReLU()
        )
        self.decoder = nn.Linear(
            # outChannels*2,
            28,
            tools.numClasses)

        self.modelPath = __file__.replace(os.getcwd(), "")[1:-3] + ".pt"

    def forward(self, inp):
        presInp = inp[0]
        flowInp = inp[1]
        # print(presInp.shape)
        # print(flowInp.shape)

        presOut = self.presCNN(presInp)
        flowOut = self.flowCNN(flowInp)

        cnnOutputs = torch.cat((presOut, flowOut), 2)

        output = self.decoder(cnnOutputs)
        output = self.output(output)
        return output, None

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import networkSettings as ns


torch.manual_seed(1)


class LeakFinder(nn.Module):
    def __init__(self):
        super(LeakFinder, self).__init__()
        self.hidden = self.init_hidden()
        self.output = ns.outputFunction
        self.lstm = nn.LSTM(input_size=ns.inputSize, hidden_size=ns.hiddenSize,
                             num_layers=ns.numLSTMLayers,
                             bidirectional=False, dropout=ns.dropout)
        self.decoder = nn.Linear(ns.hiddenSize, ns.numClasses)

    def init_hidden(self, hidden=None):
        if hidden:
            self.hidden = hidden
        else:
            self.hidden = (torch.randn(ns.numLSTMLayers, 1, ns.hiddenSize),
                           torch.randn(ns.numLSTMLayers, 1, ns.hiddenSize))
        return self.hidden

    def forward(self, inp):
        output, self.hidden = self.lstm(inp, self.hidden)
        output = self.output(self.decoder(output))
        return output, self.hidden

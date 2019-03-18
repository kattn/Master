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
                            bidirectional=True, dropout=ns.dropout)
        self.decoder = nn.Linear(ns.hiddenSize*(ns.biDirectional+1),
                                 ns.numClasses)

    def init_hidden(self, hidden=None):
        if hidden:
            self.hidden = hidden
        else:
            self.hidden = (torch.ones(ns.numLSTMLayers*(ns.biDirectional+1),
                           1, ns.hiddenSize),
                           torch.ones(ns.numLSTMLayers*(ns.biDirectional+1),
                           1, ns.hiddenSize))
        return self.hidden

    def forward(self, inp):
        output, self.hidden = self.lstm(inp, self.hidden)
        output = self.decoder(output)
        output = self.output(output)
        return output, self.hidden

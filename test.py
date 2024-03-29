import torch
import torch.nn as nn

import scenarioController as sc
from models.singleGRU.singleGRU import SingleGRU
from models.singleLSTM.singleLSTM import SingleLSTM


#### TORCH TESTING

# loss = nn.CrossEntropyLoss()
# print(loss.__dict__)
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# output = loss(input, target)
# # output.backward()

# print(input)
# print(target)
# print(output)

# weights = torch.tensor([0.1, 0.2, 0.2, 0.4, 0.1])
# loss = nn.CrossEntropyLoss(weights)
# output = loss(input, target)

# print(output)

# gru = nn.GRU(5, 3, 2)
# input = torch.randn(5, 1, 5)
# h0 = torch.randn(2, 1, 3)
# output, hn = gru(input, h0)

# print(input)
# print(h0)
# print(output)
# print(hn)


# lstm = nn.LSTM(5, 3, 2)
# input = torch.randn(5, 1, 5)
# h0 = (torch.randn(2, 1, 3), torch.randn(2, 1, 3))
# output, hn = lstm(input, h0)

# print(input)
# print(h0)
# print(output)
# print(hn)

# model testing:
module = SingleGRU()
# module.load_state_dict(torch.load("models/ResultModels/Net1_2/gru_hs20_nL3.pt"))
print(module.lstm.state_dict)
import torch.nn as nn
import torch.optim as optim

numDailySeries = 3
numSensorValues = 66
windowSize = 1  # 4*24*numDailySeries

inputSize = numSensorValues*windowSize
hiddenSize = 40
numLSTMLayers = 1
dropout = 0.1
outputFunction = nn.Sigmoid()
numClasses = 1

lr = 0.003
loss = nn.BCELoss()
optimizer = optim.Adam

# Add scenario numbers to run specific scenarios
scenarios = [3, 5, 8, 9, 12, 16, 18, 21, 22, 25, 26, 29, 31, 34, 35, 38]
numScenarios = 50  # used if no specific scenarios are given
percentTestScenarios = 0.1
numEpochs = 50
seqLength = 3000

import torch.nn as nn
import torch.optim as optim

numDailySeries = 3
numSensorValues = 66
windowSize = 1  # 4*24*numDailySeries

inputSize = numSensorValues*windowSize
hiddenSize = 10
numLSTMLayers = 1
dropout = 0.1
outputFunction = nn.Softmax()
numClasses = 1

lr = 0.003
trainLossFunc = nn.BCELoss()
testLossFunc = nn.L1Loss()
optimizer = optim.Adam

# Add scenario numbers to run specific scenarios
normalizeInput = False
days150ShortLeaks = [3, 5, 8, 9, 12, 16, 18, 21, 22, 25, 26, 29, 31, 34, 35, 38]
scenarios = []
numScenarios = 5  # used if no specific scenarios are given
percentTestScenarios = 0.2
numEpochs = 10
seqLength = 200

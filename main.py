import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import scenarioController
import tools
from models.trainer import Trainer
from models.singleLSTM.singleLSTM import SingleLSTM
from models.singleGRU.singleGRU import SingleGRU
from models.dualLSTM.dualLSTM import DualLSTM
from models.cnnGRU.cnnGRU import CNNGRU
from models.cnn.cnn import CNN

dsTime = time.time()
trainingSet, testSet = scenarioController.getDataset(
    pathToScenarios=tools.scenariosFolder,
    dataStructure="c",
    percentTestScenarios=tools.percentTestScenarios,
    sequenceSize=1,
    stepSize=1
    )

loadModel = True

print("Dataset read in time", time.time() - dsTime)
print("Trainset size:", len(trainingSet), "Testset size:", len(testSet))
print("Trainging on", [x[2] for x in trainingSet])
print("Testing on", [x[2] for x in testSet])

# trainTens = trainingSet[0][0]
# testTens = trainingSet[0][1]
# print(trainTens.shape)

module = SingleGRU()
if loadModel:
    module.load_state_dict(torch.load(module.modelPath))

trainer = Trainer(
    module=module,
    optimizer=module.optimizer,
    learningRate=module.lr,
    lossFunction=module.lossFunction)

trainer.printBenchmarks(testSet)

if not loadModel:
    trainer.train(trainingSet, testSet, module.numEpochs)
    torch.save(module.state_dict(), module.modelPath)
    print("remember to rename model file if wanted")

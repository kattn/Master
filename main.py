import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import scenarioController
import tools
from models.trainer import Trainer
from models.singleLSTM.singleLSTM import SingleLSTM
from models.dualLSTM.dualLSTM import DualLSTM
from models.cnnGRU.cnnGRU import CNNGRU

dsTime = time.time()
trainingSet, testSet = scenarioController.getDataset(
    pathToScenarios=tools.scenariosFolder,
    dataStructure="s",
    percentTestScenarios=tools.percentTestScenarios)

print("Dataset read in time", time.time() - dsTime)
print("Trainset size:", len(trainingSet), "Testset size:", len(testSet))
print("Trainging on", [x[2] for x in trainingSet])
print("Testing on", [x[2] for x in testSet])

module = CNNGRU()
trainer = Trainer(
    module=module,
    optimizer=module.optimizer,
    learningRate=module.lr,
    lossFunction=module.lossFunction)

trainer.train(trainingSet, testSet, module.numEpochs)

torch.save(module.state_dict(), module.modelPath)
print("remember to rename model file if wanted")

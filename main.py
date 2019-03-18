import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import scenarioController
import tools
from models.trainer import Trainer
import models.singleLSTM.singleLSTM as singleLSTM

dsTime = time.time()
trainingSet, testSet = scenarioController.getDataset(
    pathToScenarios=tools.scenariosFolder,
    percentTestScenarios=tools.percentTestScenarios)

print("Dataset read in time", time.time() - dsTime)
print("Trainset size:", len(trainingSet), "Testset size:", len(testSet))
print("Trainging on", [x[2] for x in trainingSet])
print("Testing on", [x[2] for x in testSet])

module = singleLSTM.SingleLSTM()
trainer = Trainer(
    module=module,
    optimizer=singleLSTM.optimizer,
    learningRate=singleLSTM.lr,
    lossFunction=singleLSTM.lossFunction)

trainer.train(trainingSet, testSet, singleLSTM.numEpochs)

torch.save(module.state_dict(), module.path)
print("remember to rename model file if wanted")

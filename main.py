import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import scenarioController
import tools
import settings
from models.trainer import Trainer
from models.singleLSTM.singleLSTM import SingleLSTM
from models.singleGRU.singleGRU import SingleGRU
from models.dualLSTM.dualLSTM import DualLSTM
from models.cnnGRU.cnnGRU import CNNGRU
from models.cnn.cnn import CNN

dsTime = time.time()
trainingSet, testSet = scenarioController.getDataset(
    pathToScenarios=settings.scenariosFolder,
    dataStructure="s",
    percentTestScenarios=settings.percentTestScenarios,
    sequenceSize=settings.sequenceSize,
    stepSize=settings.stepSize,
    targetType="long"
    )

print("Dataset read in time", time.time() - dsTime)
print("Trainset size:", len(trainingSet), "Testset size:", len(testSet))
print("Trainging on", [x[2] for x in trainingSet])
print("Testing on", [x[2] for x in testSet])

module = SingleGRU()
if settings.loadModel:
    module.load_state_dict(torch.load(settings.modelPath))

trainer = Trainer(
    module=module,
    optimizer=module.optimizer,
    learningRate=module.lr,
    lossFunction=module.lossFunction)

if not settings.loadModel:
    trainer.train(trainingSet, testSet, module.numEpochs)
    torch.save(module.state_dict(), module.modelPath)
    print("remember to rename model file if wanted")

trainer.printBenchmarks(testSet)
trainer.storeBenchmarks("benchmarks" + module.modelPath[:-3] + ".txt")
print("possible scenarios to store:")
scenarios = [x[2] for x in (trainingSet+testSet)]
print(scenarios)
scen = input("which scenario should be stored?")
while ("Scenario-"+scen) not in scenarios:
    scen = input("which scenario should be stored?")
scenarioToBeStored = [x for x in (trainingSet+testSet) if x[2] == "Scenario-" + scen][0]
trainer.storePrediction(scenarioToBeStored, "prediction.txt")

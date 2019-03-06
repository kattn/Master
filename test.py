import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import networkSettings as ns
from lstmLeakFinder import LeakFinder
import tools

plt.ion()
fig, axes = plt.subplots(2, 1)
plt.pause(0.0004)

dsTime = time.time()
trainingSet, testSet = tools.getDataset(percentTestScenarios=ns.percentTestScenarios)
print("Dataset read in time", time.time() - dsTime)
print("Trainset size:", len(trainingSet), "Testset size:", len(testSet))
print("Trainging on", [x[2] for x in trainingSet])
print("Testing on", [x[2] for x in testSet])

model = LeakFinder()

optimizer = ns.optimizer(model.parameters(), lr=ns.lr)

trainLoss = []
testLoss = []
testAccuracy = []

trainigTime = time.time()
for epoch in range(ns.numEpochs):
    print("Epoch:", epoch)
    model.train()
    for tensor, target, scenario in trainingSet:
        # So large single data set, want learning more often
        model.init_hidden()

        # mark the start of new scenario and hidden
        initHiddenMarker = len(trainLoss)
        axes[0].axvline(x=initHiddenMarker, linestyle=":")

        for tensor, target in zip(torch.split(tensor, ns.seqLength),
                                  torch.split(target, ns.seqLength)):
            output, _ = model(tensor)

            loss = ns.trainLossFunc(output, target)
            loss.backward(retain_graph=True)
            optimizer.step()
            trainLoss.append(loss.item())

        print(scenario, "trainLoss:", ns.trainLossFunc(model(tensor)[0], target))
        optimizer.zero_grad()

    model.eval()
    for tensor, target, scenario in testSet:
        model.init_hidden()
        model.zero_grad()

        initHiddenMarker = len(testLoss)
        axes[1].axvline(x=initHiddenMarker, linestyle=":")

        for tensor, target in zip(torch.split(tensor, ns.seqLength),
                                  torch.split(target, ns.seqLength)):
            output, _ = model(tensor)

            testLoss.append(ns.testLossFunc(output, target).item())
        print("testloss", ns.testLossFunc(model(tensor)[0], target).item())
        print(scenario, "testLoss:", testLoss[-1])

    # ploting
    axes[0].plot(trainLoss)
    axes[0].set_ylabel("Training loss(BCE)")
    axes[1].plot(testLoss)
    axes[1].set_ylabel("Test loss(L1Loss)")
    plt.draw()
    plt.pause(0.0004)

print("total training time:", time.time() - trainigTime)
plt.ioff()
plt.show()

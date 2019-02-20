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

model = LeakFinder()

optimizer = ns.optimizer(model.parameters(), lr=ns.lr)
lossFunc = ns.loss
testLossFunc = nn.L1Loss()

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

            if output.min() < 0:
                print(tensor)
                print(output)

            loss = lossFunc(output, target)
            loss.backward(retain_graph=True)
            trainLoss.append(loss.item())
            # for p, n in zip(model.model[-1].parameters(), model.model[-1]._all_weights[0]):
            #     if n[:6] == 'weight':
            #         print('===========\ngradient:{}\n----------\n{}'.format(n,p.grad))
            # if trainLoss[-1] < 0.32:
            #     exit()
            optimizer.step()
            optimizer.zero_grad()
        print(scenario, "trainLoss:", testLossFunc(model(tensor)[0], target))

    model.eval()
    for tensor, target, scenario in testSet:
        model.init_hidden()

        model.zero_grad()

        output, _ = model(tensor)

        if output.min() < 0:
                print(tensor)
                print(output)

        testLoss.append(testLossFunc(output, target).item())
        print(scenario, "testLoss:", testLoss[-1])

    # ploting
    axes[0].plot(trainLoss)
    axes[0].set_ylabel("Training loss(BCE)")
    axes[1].plot(testLoss)
    axes[1].set_ylabel("Test loss(L1Loss)")
    plt.draw()
    plt.pause(0.0004)

plt.ioff()
plt.show()
print("total training time:", time.time() - trainigTime)

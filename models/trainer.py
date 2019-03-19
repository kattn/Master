import time
import matplotlib.pyplot as plt
import torch


class Trainer():
    """
    Trainer ment to train and test arbitrary pytorch modules.
    """

    def __init__(self, module, optimizer, learningRate, lossFunction):
        """
        Takes the module to train, and the optimizer, learning rate, loss
        function to use for the training.
        """

        self.module = module
        self.optimizer = optimizer(self.module.parameters(), learningRate)
        self.lossFunction = lossFunction

        self.trainingDataPoints = []
        self.testDataPoints = []

        self.currentlyTraining = False

        plt.ion()
        self.fig, (self.trainingAxis, self.testAxis) = plt.subplots(2, 1)
        self.trainingAxis.set_ylabel("Training loss")
        self.testAxis.set_ylabel("Test loss")
        plt.pause(0.0004)

    def train(self, trainingSet, testSet, numEpochs):
        """
        Trains and tests the module using the given training and testing set,
        and sets the module to eval mode after training. Prints total training
        time, and time can be printed during training using
        printTrainingTime(). Plots training and test results during
        training/testing.
        """

        self.currentlyTraining, self.trainingTime = True, time.time()

        for epoch in range(numEpochs):
            print("Epoch:", epoch)

            self.trainModule(trainingSet)

            self.trainingAxis.plot(self.trainingDataPoints)
            plt.draw()
            plt.pause(0.0004)

            self.testModule(testSet)

            self.testAxis.plot(self.testDataPoints)
            plt.draw()
            plt.pause(0.0004)

        self.currentlyTraining = False
        self.printTrainingTime(identifier="Done training:")

        plt.ioff()
        plt.show()

    def trainModule(self, trainingSet):
        self.module.train()
        for tensor, target, scenario in trainingSet:
            self.module.init_hidden()
            self.optimizer.zero_grad()

            output, _ = self.module(tensor)

            loss = self.lossFunction(output, target)
            loss.backward(retain_graph=True)
            self.optimizer.step()

            self.trainingDataPoints.append(loss.item())
            print(scenario, "trainLoss:", self.trainingDataPoints[-1])

        self.module.eval()  # So the module is defaultly in eval mode

    def testModule(self, testSet):
        self.module.eval()
        for tensor, target, scenario in testSet:
            self.module.init_hidden()
            self.module.zero_grad()

            output, _ = self.module(tensor)

            self.testDataPoints.append(
                torch.sum(torch.abs(output - target)))
            print(scenario, "testLoss:", self.testDataPoints[-1])

    def printTrainingTime(self, identifier):
        """Prints how long the module has been training, with a given
        identifier"""

        if self.currentlyTraining:
            print(identifier, time.time() - self.trainingTime)

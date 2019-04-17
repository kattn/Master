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

    def trainModule(self, data):
        self.module.train()
        for trainingSet, targetSet, scenario in data:
            scenarioError = []
            if hasattr(self.module, 'init_hidden'):
                self.module.init_hidden()

            prevTime = 0
            for tensor, target in zip(trainingSet, targetSet):
                self.module.zero_grad()

                output = self.module(tensor)

                loss = self.lossFunction(output, target)
                loss.backward(retain_graph=True)
                self.optimizer.step()

                scenarioError.append(
                    torch.mean(torch.abs(output - target)).item()
                )

            self.trainingDataPoints.append(
               sum(scenarioError)/len(scenarioError))

            print(
                "Training", scenario,
                "Accuracy", self.trainingDataPoints[-1]
                )

        self.module.eval()  # So the module is defaultly in eval mode

    def testModule(self, data):
        self.module.eval()
        for testSet, targetSet, scenario in data:
            scenarioError = []
            if hasattr(self.module, 'init_hidden'):
                self.module.init_hidden()

            for tensor, target in zip(testSet, targetSet):
                self.module.zero_grad()

                output = self.module(tensor)

                scenarioError.append(
                    torch.mean(torch.abs(output - target)).item()
                    )

            self.testDataPoints.append(
                sum(scenarioError)/len(scenarioError))
            print(
                "Testing", scenario, "Accuracy",
                self.testDataPoints[-1])

    def printBenchmarks(self, data, ge=0.5):
        """Takes a data input of tensors and targets. ge is the value that is
        used to treshold the output to 1 and 0.
        Returns TPR, FPR and accuracy for each data"""
        self.module.eval()
        for tensors, targets, scenario in data:
            tp = 0
            fp = 0
            tn = 0
            p = 0
            n = 0
            for tensor, target in zip(tensors, targets):
                output = self.module(tensor).ge(ge)
                # uses ge to create a binary output vector

                for x, y in zip(output.squeeze(), target.squeeze()):
                    x = int(x)
                    y = int(y)

                    if y == 1:
                        p += 1
                    else:
                        n += 1

                    if x == y == 1:
                        tp += 1
                    if x == 1 and y == 0:
                        fp += 1
                    if x == y == 0:
                        tn += 1

            acc = (tp+tn)/(p+n)
            if p == 0:
                tpr = "-"
            else:
                tpr = tp/p
            if n == 0:
                fpr = "-"
            else:
                fpr = fp/n

            print(
                scenario,
                "Accuracy", acc,
                "TPR", tpr,
                "FPR", fpr,
                "p", p)

    def printTrainingTime(self, identifier):
        """Prints how long the module has been training, with a given
        identifier"""

        if self.currentlyTraining:
            print(identifier, time.time() - self.trainingTime)

import time
import matplotlib.pyplot as plt
import torch
import random

from benchmark import Benchmark
import settings


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

        self.benchmarks = {}

        plt.ion()
        self.fig, (self.trainingAxis, self.testAxis, self.epochAxis) = plt.subplots(3, 1)
        self.fig.canvas.set_window_title(module.name + " " + module.modelPath[:-3])
        plt.pause(0.0004)

    def train(self, trainingSet, testSet, numEpochs):
        """
        Trains and tests the module using the given training and testing set,
        and sets the module to eval mode after training. Prints total training
        time, and time can be printed during training using
        printTrainingTime(). Plots training and test results during
        training/testing.
        """

        trainingTime = []
        trainingEpochError = []
        testEpochError = []

        while numEpochs != 0:
            for epoch in range(numEpochs):
                print("Epoch:", epoch)
                trainingTime.append(time.time())

                random.shuffle(trainingSet)
                self.trainModule(trainingSet)
                # self.storePrediction(trainingSet[0], "debug.txt")
                # input()

                # As the size of each scenario is the same, it is ok to avrage
                # the avrages to get a total average
                epochSlice = self.trainingDataPoints[-len(trainingSet):]
                trainingEpochError.append(sum(epochSlice)/len(epochSlice))

                self.testModule(testSet)

                epochSlice = self.testDataPoints[-len(trainingSet):]
                testEpochError.append(sum(epochSlice)/len(epochSlice))

                self.updatePlot(trainingEpochError, testEpochError)

            trainingTime[-1] = time.time() - trainingTime[-1]
            print(self.module.modelPath)
            print(self.module.name)
            self.printBenchmarks(trainingSet+testSet)

            numEpochs = int(input("train more(0 or number of epochs):"))

        print("total training time:", sum(trainingTime))

        # fName = "Error_" + self.module.name + ".txt"
        # with open(fName, 'w+') as f:
        #     f.write("training, test")
        #     for trainErr, testErr in zip(trainingEpochError, testEpochError):
        #         f.write(f"{trainErr:.4f} \t {testErr:.4f}\n")

        plt.ioff()
        plt.show()

    def trainModule(self, data):
        self.module.train()
        for trainingSet, targetSet, scenario in data:
            scenarioError = []
            if hasattr(self.module, 'init_hidden'):
                self.module.init_hidden()

            for tensor, target in zip(trainingSet, targetSet):
                self.module.zero_grad()

                output = self.module(tensor)
                # output = torch.cat((output, 1-output), 2)

                loss = self.lossFunction(output.squeeze(1), target.squeeze())
                loss.backward(retain_graph=True)
                self.optimizer.step()
                # for p,n in zip(self.module.gru.parameters(),self.module.gru._all_weights[0]):
                #     if n[:6] == 'weight':
                #         print('===========\ngradient:{}\n----------\n{}'.format(n,p.grad))

                # convert as target may be long when using CrossEntropy
                if target.dtype is torch.long:
                    target = target.float()

                scenarioError.append(
                    torch.mean(torch.abs(output - target)).item()
                )

            self.trainingDataPoints.append(
               sum(scenarioError)/len(scenarioError))

            print(
                "Training", scenario,
                "Error", self.trainingDataPoints[-1]
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
                    torch.mean(torch.abs(output - target.float())).item()
                    )

            self.testDataPoints.append(
                sum(scenarioError)/len(scenarioError))
            print(
                "Testing", scenario, "Error",
                self.testDataPoints[-1])

    def updatePlot(self, trainingEpochError, testEpochError):
        self.trainingAxis.clear()
        self.testAxis.clear()
        self.epochAxis.clear()

        self.trainingAxis.set_ylabel("Training Error")
        self.testAxis.set_ylabel("Test Error")
        self.epochAxis.set_ylabel("Avg epoch error")

        self.trainingAxis.plot(self.trainingDataPoints)
        self.testAxis.plot(self.testDataPoints)
        self.epochAxis.plot(trainingEpochError, label="Training Epoch Error")
        self.epochAxis.plot(testEpochError, label="Test Epoch Error")
        self.epochAxis.legend()

        plt.pause(0.0004)

    def printBenchmarks(self, data):
        """Takes a data input of tensors and targets. ge is the value that is
        used to treshold the output to 1 and 0.
        Returns TPR, FPR and accuracy for each data"""
        self.module.eval()

        print("Name" + "\t\t" + "Leak" + "\t" + "TPR" + "\t" + "FPR" + "\t" + "Accuracy")
        for tensors, targets, scenario in data:
            bench = Benchmark()

            for tensor, target in zip(tensors, targets):
                classification = self.module.classify(tensor)[1]

                bench += Benchmark(
                    classification.squeeze(), target.squeeze())

            if bench.p != 0:
                leak = "yes"
            else:
                leak = "no"
            print(
                f"{scenario:13}\t{leak}\t{bench.getTPR():.3f}\t{bench.getFPR():.3f}\t{bench.getAccuracy():.3f}"
            )
            self.benchmarks[scenario] = bench

    def storeBenchmarks(self, path):
        with open(path, 'w+') as f:
            f.write(
                "Name" + "\t\t\t" + "Leak" + "\t" + "TPR" + "\t\t" +
                "FPR" + "\t\t" + "Acc" + "\t\t" + "DT" + "\n")
            for scenario in sorted(self.benchmarks.keys()):
                bench = self.benchmarks[scenario]
                if bench.p != 0:
                    leak = "yes"
                else:
                    leak = "no"
                f.write(f"{scenario:13}\t{leak}\t\t{bench.getTPR():.3f}\t{bench.getFPR():.3f}\t{bench.getAccuracy():.3f}\t{bench.dt}\n")

    def storePrediction(self, scenario, path):
        self.module.eval()
        (tensors, targets, scenario) = scenario

        with open(path, 'w+') as f:
            f.write("Pred \t\t label \t target\n")
            for tensor, target in zip(tensors, targets):
                output, classification = self.module.classify(tensor)

                if settings.singleTargetValue:
                    line = f"{output}\t{binariesed.item()}\t{target.item()}\n"
                    f.write(line)
                else:
                    for x, b, y in zip(output.squeeze(), classification.squeeze(), target.squeeze()):
                        line = f"{x}" + "\t\t" + str(b.item()) + "\t\t" + str(y.item()) + "\n"
                        f.write(line)

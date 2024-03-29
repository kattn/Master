import pandas
import torch
from os import listdir
from os.path import isfile, join
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random

import tools
import settings
from os import scandir


class ScenarioController:

    def __init__(self, pathToScenario, readPressures=True, readFlows=False,
                 readDemands=False, readLabels=True):

        if(pathToScenario[-1] != "/"):
            pathToScenario += "/"
        self.pathToScenario = pathToScenario

        self.pressures = None
        self.flows = None
        self.demands = None
        self.labels = None
        if readPressures:
            self.pressures = self.readPressures()
        if readFlows:
            self.flows = self.readFlows()
        if readDemands:
            self.demands = self.readDemands()
        if readLabels:
            self.labels = self.readLabels()

    def readPressures(self):
        self.pressures = tools.readCVSFolder(
            self.pathToScenario + "Pressures/")
        self.pressures.set_index("Timestamp", drop=True, inplace=True)
        return self.pressures

    def readDemands(self):
        self.demands = tools.readCVSFolder(self.pathToScenario + "Demands/")
        self.demands.set_index("Timestamp", drop=True, inplace=True)
        return self.demands

    def readFlows(self):
        self.flows = tools.readCVSFolder(self.pathToScenario + "Flows/")
        self.flows.set_index("Timestamp", drop=True, inplace=True)
        return self.flows

    def readLabels(self):
        self.labels = pandas.read_csv(
            self.pathToScenario + "Labels.csv",
            names=["Timestamp", "Label"], header=0)
        self.labels.set_index("Timestamp", drop=True, inplace=True)
        return self.labels

    def getPressures(self, withLabels=True):
        if self.pressures is None:
            self.readPressures()

        if withLabels:
            labels = self.readLabels()
            return self.pressures.merge(labels, on="Timestamp")

        return self.pressures

    def getFlows(self, withLabels=True):
        if self.flows is None:
            self.readFlows()

        if withLabels:
            labels = self.readLabels()
            return self.flows.merge(labels, on="Timestamp")

        return self.flows

    def getDemands(self, withLabels=True):
        if self.demands is None:
            self.readDemands()

        if withLabels:
            labels = self.readLabels()
            return self.demands.merge(labels, on="Timestamp")

        return self.demands

    def getLabels(self):
        if self.labels is None:
            return self.labels
        else:
            return self.readLabels()

    def getAllData(self, withLabels=True):

        dfP = self.getPressures(withLabels)
        dfF = self.getFlows(False)

        df = dfP.merge(dfF, on="Timestamp")

        return df

    def plotTimeInterval(self, start, stop, normalize=False):
        # slize measures to plot
        pressureSlize = self.getPressures(withLabels=False).loc[start:stop, :]
        flowsSlize = self.getFlows(withLabels=False).loc[start:stop]
        # demandsSlize = self.getDemands(withLabels=False).loc[start:stop, :]
        labelSlize = self.getLabels().loc[start:stop]

        if normalize:
            pressureSlize = pressureSlize.apply(normalizeDf, axis=1)
            flowsSlize = flowsSlize.apply(normalizeDf, axis=1)

        # plot
        fig, ax = plt.subplots(3, sharex=True)

        # dax = demandsSlize.plot(ax=ax[0])
        # dax.set_ylabel("Demand")
        # dax.get_legend().set_visible(False)
        pax = pressureSlize.plot(ax=ax[0])
        pax.set_ylabel("Pressure")
        pax.get_legend().set_visible(False)
        fax = flowsSlize.plot(ax=ax[1])
        fax.set_ylabel("Flow")
        fax.get_legend().set_visible(False)
        lax = labelSlize.plot(ax=ax[2])
        lax.set_ylabel("Label")
        lax.get_legend().set_visible(False)

        # prettify plot
        fig.subplots_adjust(hspace=0)
        handles, labels = pax.get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="center left",
            bbox_to_anchor=(0.6, 0.5), ncol=2)
        ticklabels = pressureSlize.index
        lax.xaxis.set_major_formatter(ticker.IndexFormatter(ticklabels))
        pax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.4)
        plt.tight_layout(rect=[0, 0, 0.6, 1])

        plt.show()


def normalizeDf(dfRow):
    df = (dfRow - dfRow.min())/(dfRow.max()-dfRow.min())
    return df


def getDataset(
        pathToScenarios,
        dataStructure,
        numTestScenarios=0,
        percentTestScenarios=0.2,
        sequenceSize=1,
        stepSize=1,
        targetType="float",
        sensors=None):
    """
    The data structure parameter structures the input accordingly:
    c -- combines the flow and pressure measurements
    s -- seperates the flow and pressure measurements into tuples

    SequenceSize and stepSize decides how the data is split. If the input is
    invalid the sequenceSize and stepSize are set to default values of 1 and 1.
    NB: Does not check if the sequenceSize or stepSize is greater the the size
    of the dataset.
    """

    data = []

    if sensors is not None:
        sensors = ["Node_"+str(sensor) for sensor in sensors]

    # Decide if specific scenarios or just a number of them
    if settings.lockDataSets:
        scenarios = settings.train + settings.test
        count = len(scenarios)
    elif len(settings.scenarios) != 0:
        scenarios = settings.scenarios
        count = len(scenarios)
    else:
        count = settings.numScenarios

    # Ensure that sequenceSize and stepSize are valid values
    if sequenceSize < 1:
        sequenceSize = 1
    if stepSize < 1:
        sequenceSize = 1

    for dirEntry in scandir(pathToScenarios):
        if dirEntry.is_dir():

            # Decide if specific scenarios or just a number of them
            if len(settings.scenarios) != 0 or settings.lockDataSets:
                if dirEntry.path.split("/")[-1] not in scenarios:
                    continue
            else:
                if count == 0:
                    break

            sc = ScenarioController(dirEntry.path)

            # Read the target tensors
            dfLabel = sc.getLabels()
            numColumnsTarget = settings.numClasses
            if targetType == "float":
                target = torch.tensor(dfLabel["Label"].values, dtype=torch.float32)
            elif targetType == "long":
                target = torch.tensor(dfLabel["Label"].values, dtype=torch.long)

            if stepSize != 1 or sequenceSize != 1:
                target = target.unfold(0, sequenceSize, stepSize)
            else:
                target = target.unsqueeze(0)
            target = target.unsqueeze(2).unsqueeze(2)

            # Structure the target tensor
            if settings.singleTargetValue:
                target = target.max(1, keepdim=True)[0]

            # Read the feature tensors
            if dataStructure == "c":
                df = sc.getAllData()
                if sensors:
                    df = df[sensors]
                numColumnsTensor = tools.getNumSensors("t")

                # Convert the dataframe to a tensor
                inp = torch.tensor(
                    df.loc[:, df.columns.difference(
                        ["Label", "Timestamp"])].values, dtype=torch.float32)

                if settings.normalizeInput:
                    inp = tools.normalizeWindow(inp, sequenceSize)

                if stepSize != 1 or sequenceSize != 1:
                    inp = inp.unfold(0, sequenceSize, stepSize).transpose(1, 2)
                else:
                    inp = inp.unsqueeze(0)

                inp = inp.unsqueeze(2)

            elif dataStructure == "s":
                dfPressure = sc.getPressures(False)
                if sensors:
                    dfPressure = dfPressure[sensors]
                dfFlow = sc.getFlows(False)

                # Convert the dataframes to a tensors
                presInp = torch.tensor(
                    dfPressure.loc[:, dfPressure.columns.difference(
                        ["Label", "Timestamp"])].values, dtype=torch.float32)

                flowInp = torch.tensor(
                    dfFlow.loc[:, dfFlow.columns.difference(
                        ["Label", "Timestamp"])].values, dtype=torch.float32)

                if settings.normalizeInput:
                    presInp = tools.normalizeWindow(presInp, sequenceSize)
                    flowInp = tools.normalizeWindow(flowInp, sequenceSize)

                if stepSize != 1 or sequenceSize != 1:
                    presInp = presInp.unfold(0, sequenceSize, stepSize).transpose(1, 2)
                    flowInp = flowInp.unfold(0, sequenceSize, stepSize).transpose(1, 2)
                else:
                    presInp = torch.unsqueeze(presInp, 0)
                    flowInp = torch.unsqueeze(flowInp, 0)

                presInp = presInp.unsqueeze(2)
                flowInp = flowInp.unsqueeze(2)

                inp = [(torch.tensor(pres), torch.tensor(flow)) for pres, flow in zip(presInp.tolist(), flowInp.tolist())]

            data.append((inp, target, dirEntry.path.split("/")[-1]))
            # print(count, "files on the wall,", count, "files to read")
            # print(
            #    "Take one down, pass it around,", count-1, "files on the wall")
            count -= 1

    numTests = 0
    if numTestScenarios != 0:
        numTests = numTestScenarios
    if percentTestScenarios != 0.0:
        numTests = int(len(data)*percentTestScenarios)

    if settings.lockDataSets:
        trainingSet = []
        testSet = []
        for scenario in data:
            if scenario[2] in settings.train:
                trainingSet.append(scenario)
            else:
                testSet.append(scenario)
    else:
        random.shuffle(data)
        trainingSet = data[numTests:]
        testSet = data[:numTests]

    return trainingSet, testSet


# testing
if __name__ == "__main__":
    sc = ScenarioController(
        "NetworkModels/Benchmarks/Hanoi_CMH/Scenario-77", readFlows=False, readPressures=False)
    sc.plotTimeInterval("2017-01-27 00:00:00", "2017-01-30 23:45:00", True)
    sc.plotTimeInterval("2017-01-27 00:00:00", "2017-01-30 23:45:00", False)
    # df = sc.getAllData()
    # columns = ["Node_"+str(sensor) for sensor in [12, 21]]
    # print(df[columns])

    # trainingSet, testSet = getDataset(
    #     pathToScenarios=settings.scenariosFolder,
    #     dataStructure="s",
    #     percentTestScenarios=settings.percentTestScenarios,
    #     sequenceSize=settings.sequenceSize,
    #     stepSize=settings.stepSize,
    #     targetType="long",
    #     sensors=[12,21]
    #     )

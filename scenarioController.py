import pandas
import torch
from os import listdir
from os.path import isfile, join
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random

import tools
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

    def plotTimeInterval(self, start, stop):
        # slize measures to plot
        pressureSlize = self.getPressures(withLabels=False).loc[start:stop, :]
        flowsSlize = self.getFlows(withLabels=False).loc[start:stop]
        demandsSlize = self.getDemands(withLabels=False).loc[start:stop, :]
        labelSlize = self.getLabels().loc[start:stop]

        # plot
        fig, ax = plt.subplots(4, sharex=True)

        dax = demandsSlize.plot(ax=ax[0])
        dax.set_ylabel("Demand")
        dax.get_legend().set_visible(False)
        pax = pressureSlize.plot(ax=ax[1])
        pax.set_ylabel("Pressure")
        pax.get_legend().set_visible(False)
        fax = flowsSlize.plot(ax=ax[2])
        fax.set_ylabel("Flow")
        fax.get_legend().set_visible(False)
        lax = labelSlize.plot(ax=ax[3])
        lax.set_ylabel("Label")
        lax.get_legend().set_visible(False)

        # prettify plot
        fig.subplots_adjust(hspace=0)
        handles, labels = dax.get_legend_handles_labels()
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


def getDataset(
        pathToScenarios,
        dataStructure,
        numTestScenarios=0,
        percentTestScenarios=0.2):
    """
    The data structure parameter structures the input accordingly:
    c -- combines the flow and pressure measurements
    s -- seperates the flow and pressure measurements into tuples
    """

    data = []

    # Decide if specific scenarios or just a number of them
    if len(tools.scenarios) != 0:
        scenarios = ["Scenario-" + str(x) for x in tools.scenarios]
        count = len(tools.scenarios)
    else:
        count = tools.numScenarios

    for dirEntry in scandir(pathToScenarios):
        if dirEntry.is_dir():

            # Decide if specific scenarios or just a number of them
            if len(tools.scenarios) != 0:
                if dirEntry.path.split("/")[-1] not in scenarios:
                    continue
            else:
                if count == 0:
                    break

            sc = ScenarioController(dirEntry.path)

            dfLabel = sc.getLabels()
            numColumnsTarget = tools.numClasses
            target = torch.tensor(dfLabel["Label"].values, dtype=torch.float32)
            target = target.view(-1, 1, numColumnsTarget)
            target = torch.unsqueeze(target, 1)

            if dataStructure == "c":
                df = sc.getAllData()
                numColumnsTensor = tools.getNumSensors("t")
                inp = torch.tensor(
                    df.loc[:, df.columns.difference(
                        ["Label", "Timestamp"])].values, dtype=torch.float32)
                if tools.normalizeInput:
                    inp = tools.normalize(
                        inp.view(-1, 1, numColumnsTensor), p=1, dim=2)
                else:
                    inp = inp.view(-1, numColumnsTensor)
                inp = torch.unsqueeze(inp, 1)

            elif dataStructure == "s":
                dfPressure = sc.getPressures(False)
                dfFlow = sc.getFlows(False)
                numPresSensor = tools.getNumSensors("p")
                numFlowSensor = tools.getNumSensors("f")

                presInp = torch.tensor(
                    dfPressure.loc[:, dfPressure.columns.difference(
                        ["Label", "Timestamp"])].values, dtype=torch.float32)
                flowInp = torch.tensor(
                    dfFlow.loc[:, dfFlow.columns.difference(
                        ["Label", "Timestamp"])].values, dtype=torch.float32)
                if tools.normalizeInput:
                    presInp = torch.unsqueeze(tools.normalize(
                        presInp.view(-1, 1, numPresSensor), p=1, dim=2), 1)
                    flowInp = torch.unsqueeze(tools.normalize(
                        flowInp.view(-1, 1, numFlowSensor), p=1, dim=2), 1)
                else:
                    presInp = torch.unsqueeze(
                        presInp.view(-1, 1, numPresSensor), 1)
                    flowInp = torch.unsqueeze(
                        flowInp.view(-1, 1, numFlowSensor), 1)
                inp = [(torch.tensor(pres), torch.tensor(flow)) for pres, flow in zip(presInp.tolist(), flowInp.tolist())]
            data.append((inp, target, dirEntry.path.split("/")[-1]))
            print(count, "files on the wall,", count, "files to read")
            print(
                "Take one down, pass it around,", count-1, "files on the wall")
            count -= 1

    numTests = 0
    if numTestScenarios != 0:
        numTests = numTestScenarios
    if percentTestScenarios != 0.0:
        numTests = int(len(data)*percentTestScenarios)

    random.shuffle(data)
    trainingSet = data[numTests:]
    testSet = data[:numTests]
    return trainingSet, testSet


# testing
if __name__ == "__main__":
    sc = ScenarioController(
        "NetworkModels/Benchmarks/Net1/Scenario-91", readPressures=False)
    sc.plotTimeInterval("2017-01-01 00:00:00", "2017-01-28 10:45:00")

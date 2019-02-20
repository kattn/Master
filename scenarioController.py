import pandas
from os import listdir
from os.path import isfile, join
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tools


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
        self.pressures = tools.readCVSFolder(self.pathToScenario + "Pressures/")
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
        self.labels = pandas.read_csv(self.pathToScenario + "Labels.csv", names=["Timestamp", "Label"], header=0)
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
        pressureSlize = self.getPressures(withLabels=False).loc[start:stop, "Node_1":]
        flowsSlize = self.getFlows(withLabels=False).loc[start:stop]
        demandsSlize = self.getDemands(withLabels=False).loc[start:stop, "Node_1":]
        labelSlize = self.getLabels().loc[start:stop]

        # plot
        fig, ax = plt.subplots(4, sharex=True)

        pax = pressureSlize.plot(ax=ax[0])
        pax.set_ylabel("Pressure")
        pax.get_legend().set_visible(False)
        fax = flowsSlize.plot(ax=ax[1])
        fax.set_ylabel("Flow")
        fax.get_legend().set_visible(False)
        dax = demandsSlize.plot(ax=ax[2])
        dax.set_ylabel("Demand")
        dax.get_legend().set_visible(False)
        lax = labelSlize.plot(ax=ax[3])
        lax.set_ylabel("Label")
        lax.get_legend().set_visible(False)

        # prettify plot
        fig.subplots_adjust(hspace=0)
        handles, labels = dax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.6, 0.5), ncol=2)
        ticklabels = pressureSlize.index
        pax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
        pax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.4)
        plt.tight_layout(rect=[0, 0, 0.6, 1])

        plt.show()


# testing
if __name__ == "__main__":
    sc = ScenarioController("NetworkModels/Benchmarks/Hanoi_CMH/Scenario-4", readPressures=False)
    sc.plotTimeInterval("2017-02-19 00:00:00", "2017-02-21 00:00:00")

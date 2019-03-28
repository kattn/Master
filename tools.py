import pandas
import torch
from torch.nn.functional import normalize
from os import scandir, listdir
from natsort import natsorted
import matplotlib.pyplot as plt
import random
import wntr

numClasses = 1

normalizeInput = True
days150ShortLeaks = [3, 5, 8, 9, 12, 16, 18, 21, 22, 25, 26, 29, 31, 34, 35, 38]
scenarios = []
numScenarios = 50  # used if no specific scenarios are given
percentTestScenarios = 0.5
network = "Net1"
scenariosFolder = "NetworkModels/Benchmarks/" + network + "/"
inpFile = "NetworkModels/networks/" + network + ".inp"


def readCVSFolder(path):
    csvs = [entry.path for entry in scandir(path) if entry.is_file]
    csvs = natsorted(csvs)

    df = pandas.read_csv(csvs[0], names=["Timestamp", "Node_0"], header=0)
    for index, csv in enumerate(csvs[1:], 1):
        df = df.merge(
            pandas.read_csv(
                csv,
                names=["Timestamp", "Node_"+str(index)],
                header=0),
            on="Timestamp")
    return df


def normalizeWindow(tensor, windowSize):
    """Normalizes each line based on the max within a given window size
    and removes the beginning and trailing data that is outside a full
    size window"""
    for i in range(windowSize, tensor.shape[0]-windowSize):
        tensor[i, :, :] = torch.div(
            tensor[i, :, :],
            (max(1e-12, tensor[i-windowSize:i+windowSize, :, :].abs().max())))

    return tensor[windowSize:tensor.shape[0]-windowSize, :, :]


def drawWDN(inpFile):
    wn = wntr.network.WaterNetworkModel(inpFile)
    wntr.graphics.plot_network(wn, title=wn.name)
    plt.show()


def getNumSensors(sensorType):
    """ Returns number of sensors of given type.
    "t" -- if total number is wanted
    "p" -- if pressure is wanted
    "f" -- if flow is wanted

    Calculates number by counting the number of files in the pressure/flow
    measurements folders.
    """

    if sensorType == "f":
        path = scenariosFolder + "Scenario-1/Flows"
        paths = [entry.path for entry in scandir(path) if entry.is_file]
        return len(paths)
    elif sensorType == "p":
        path = scenariosFolder + "Scenario-1/Pressures"
        paths = [entry.path for entry in scandir(path) if entry.is_file]
        return len(paths)
    elif sensorType == "t":
        return getNumSensors("f") + getNumSensors("p")
    else:
        raise Exception("Invalid sensorType " + sensorType)

if __name__ == "__main__":
    drawWDN(tools.inpFile)

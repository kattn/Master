import torch
import pandas
from torch.nn.functional import normalize
from os import scandir, listdir
from natsort import natsorted
import matplotlib.pyplot as plt
import random
import wntr

import settings


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
    """Normalizes values of the tensor between 0 and 1 along dimention 0 based
    on the max within the previous windowSized data using padding on the
    beginnning"""
    for i in range(1, tensor.shape[0]):
        tensor[i-1:i, :].div_(
            max(1e-12, tensor[max(0, i-windowSize):i, :].abs().max())
            ).abs_()
    return tensor


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
        path = settings.scenariosFolder + "Scenario-1/Flows"
        paths = [entry.path for entry in scandir(path) if entry.is_file]
        return len(paths)
    elif sensorType == "p":
        path = settings.scenariosFolder + "Scenario-1/Pressures"
        paths = [entry.path for entry in scandir(path) if entry.is_file]
        return len(paths)
    elif sensorType == "t":
        return getNumSensors("f") + getNumSensors("p")
    else:
        raise Exception("Invalid sensorType " + sensorType)

if __name__ == "__main__":
    drawWDN(inpFile)

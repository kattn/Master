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

    df = pandas.read_csv(csvs[0], names=["Timestamp", csvs[0].split("/")[-1][:-4]], header=0)
    for index, csv in enumerate(csvs[1:], 1):
        df = df.merge(
            pandas.read_csv(
                csv,
                names=["Timestamp", csv.split("/")[-1][:-4]],
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
    wntr.graphics.plot_network(wn, title=wn.name, node_labels=True)
    plt.show()


def getNumSensors(sensorType):
    """ Returns number of sensors of given type.
    "t" -- if total number is wanted
    "p" -- if pressure is wanted
    "f" -- if flow is wanted

    Either returns the number of sensors in settings, oralculates number by
    counting the number of files in the pressure/flow measurements folders.
    """
    if settings.sensors is not None:
        return len(settings.sensors)

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


def storeInputOutputValues(inp, output, path="ioExample.txt", format="asci"):
    """Stores input and output tensors in specified format to file specified
    by path"""

    with open(path, "w+") as f:
        timesteps = inp.squeeze().t().tolist()
        output = [round(x, 3) for x in output.squeeze().tolist()]

        for sensors in timesteps:
            sens = [round(x, 3) for x in sensors]
            f.write(str(sens) + "\n")
        f.write(str(output) + "\n")


def printLeakStats(path, scens=None, leakDim=False, numLeakLabels=False):
    """
    Takes path to the scenarios folder and scenarios to print.
    If no scenarios are given, reads the whole folder.
    """
    if scens is None:
        paths = [entry.path for entry in scandir(path)]
    else:
        scens = [str(scen) for scen in scens]
        paths = [entry.path for entry in scandir(path) if entry.path.split("-")[-1] in scens]
    leakStats = []

    # prints scenario - leakDim of every leak
    if leakDim:
        leakStats.append("Scenario \t leakDim")
        for path in paths:
            if "." in path:
                continue

            scenario = path.split("/")[-1]
            leaksFolder = path + "/Leaks/"
            for cvsFile in listdir(leaksFolder):
                if "info" in cvsFile:
                    with open(leaksFolder+cvsFile, "r") as f:
                        leakDim = f.readlines()[3].split(",")[-1].strip()
                    leakStats.append(f"{scenario} \t {float(leakDim):.3}")

    # prints total leak and non leak labels, and for only leak scenarios
    if numLeakLabels:
        df = pandas.read_csv(path+"/Labels.csv")
        if scens is not None:
            df = pandas.read_csv(path+"/Labels.csv").loc[df["Scenario"].isin(scens)]

        leakScenarios = df.loc[df["Label"] == 1.0]
        numLeakScenarios = len(leakScenarios.index)
        numNonLeakScenarios = len(df.index) - numLeakScenarios
        leakStats.append("#Leak Scenarios: " + str(numLeakScenarios))
        leakStats.append("#Non Leak Scenarios: " + str(numNonLeakScenarios))

        leakLabels = 0
        nonLeakLabels = 0
        for path in paths:
            if "." in path:
                continue

            scenLabelsDF = pandas.read_csv(path+"/Labels.csv", usecols=["Label"])
            leakLabels += len(scenLabelsDF.loc[scenLabelsDF["Label"] == 1.0].index)
            nonLeakLabels += len(scenLabelsDF.loc[scenLabelsDF["Label"] == 0.0].index)

        totalLabelsWithNonLeaks = leakLabels + nonLeakLabels
        totalLabelsWithoutNonLeaks = leakLabels + nonLeakLabels - numNonLeakScenarios*2880
        print(totalLabelsWithNonLeaks)
        print(totalLabelsWithoutNonLeaks)

        leakStats.append(f"%Leaks with non Leak: {leakLabels/totalLabelsWithNonLeaks:.3f}")
        leakStats.append(f"%Leaks without non Leak: {leakLabels/totalLabelsWithoutNonLeaks:.3f}")

    for line in leakStats:
        print(line)


if __name__ == "__main__":
    # drawWDN("NetworkModels/networks/Net1.inp")
    # drawWDN("NetworkModels/networks/Net3.inp")
    # drawWDN("NetworkModels/networks/Hanoi_CMH.inp")

    printLeakStats("NetworkModels/Benchmarks/Hanoi_CMH", scens=[185, 63, 31, 21, 169, 142, 184, 162, 138, 112, 192, 51, 99, 55, 122, 91, 67, 130, 171, 79, 171, 193, 10, 67, 157, 16, 30, 144, 86, 177, 198, 155, 40, 167, 200, 123, 176, 152, 180, 127, 163, 40, 70, 149, 12, 49, 92, 197, 28, 99], numLeakLabels=True)

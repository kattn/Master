import pandas
import torch
from torch.nn.functional import normalize
from os import scandir
from natsort import natsorted
import matplotlib.pyplot as plt
import random
import networkSettings as ns

from scenarioController import ScenarioController


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


# Normalizes each line based on the max within a given window size
# and removes the beginning and trailing data that is outside a full
# size window
def normalizeWindow(tensor, windowSize):
    for i in range(windowSize, tensor.shape[0]-windowSize):
        tensor[i, :, :] = torch.div(tensor[i, :, :],
                                    (max(1e-12,
                                     tensor[i-windowSize:i+windowSize, :, :]
                                     .abs().max())))
    return tensor[windowSize:tensor.shape[0]-windowSize, :, :]


def getDataset(pathToScenarios="NetworkModels/Benchmarks/Hanoi_CMH",
               numTestScenarios=0,
               percentTestScenarios=0.1):
    data = []

    # Decide if specific scenarios or just a number of them
    if len(ns.scenarios) != 0:
        scenarios = ["Scenario-" + str(x) for x in ns.scenarios]
        count = len(ns.scenarios)
    else:
        count = ns.numScenarios

    for dirEntry in scandir(pathToScenarios):
        if dirEntry.is_dir():

            # Decide if specific scenarios or just a number of them
            if len(ns.scenarios) != 0:
                if dirEntry.path.split("/")[-1] not in scenarios:
                    continue
            else:
                if count == 0:
                    break

            sc = ScenarioController(dirEntry.path)
            df = sc.getAllData()
            target = torch.tensor(df["Label"].values, dtype=torch.float32)
            target = target.view(-1, 1, 1)
            tensor = torch.tensor(df.loc[:, df.columns.difference(["Label", "Timestamp"])].values, dtype=torch.float32)
            if ns.normalizeInput:
                tensor = normalize(tensor.view(-1, 1, ns.numSensorValues), p=1, dim=2)
            else:
                tensor = tensor.view(-1, 1, ns.numSensorValues)
            data.append((tensor, target, dirEntry.path.split("/")[-1]))
            print(count, "files on the wall,", count, "files to read")
            print("Take one down, pass it around,", count-1, "files on the wall")
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

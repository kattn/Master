import pandas
import torch
from torch.nn.functional import normalize
from os import scandir
from natsort import natsorted
import matplotlib.pyplot as plt
import random
import networkSettings as ns
import wntr

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


def drawWDN(inpFile="NetworkModels/networks/Net3.inp"):
    wn = wntr.network.WaterNetworkModel(inpFile)
    wntr.graphics.plot_network(wn, title=wn.name)
    plt.show()


if __name__ == "__main__":
    drawWDN()

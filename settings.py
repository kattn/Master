loadModel = False
modelPath = "flow_gru_hs20_nL2.pt"

small = [65, 112, 302]
med = [55, 57, 131]
big = [46, 89, 298]
bigger = [279, 370, 70, 104]
train40 = [154, 220, 325, 240, 337, 152, 75, 235, 311, 302, 66, 22, 48, 232, 131, 331, 124, 55, 38, 141, 283, 69, 48, 16, 387, 67, 103, 6, 333, 285, 231, 148, 294, 165, 63, 330, 245, 313, 108]
trainIsh40Scens = [172, 98, 239, 331, 113, 40, 313, 348, 229, 252, 67, 148, 229, 294, 50, 195, 47, 139, 269, 389] + small + med + big + bigger
train40Big = [120, 276, 289, 219, 399, 94, 109, 250, 35, 133, 161, 114, 237, 231, 301, 199, 180, 338, 89, 193, 265, 348, 17, 207, 46, 298, 105, 305, 163, 285, 19, 148, 100, 121, 397, 257, 125, 36, 324, 364]

# hanoi train scens [112, 65, 302, 55, 57, 131, 46, 89, 298]
# hanoi train 40 scens [112, 65, 302, 55, 57, 131, 46, 89, 298]
# hanoi test scens [20, 177, 379, 13, 157, 236, 257, 324, 114, 309, 366]

trainHanoi = ["Scenario-"+str(i) for i in train40]
testHanoi = ["Scenario-"+str(i) for i in [89, 20, 177, 379, 13, 157, 236, 257, 324, 114, 309, 366]]# 7, 15, 89, 91, 112, 139, 151, 174, 182, 189]]
sensors = None  # [12, 1, 21, 17, 28]

trainNet1 = ["Scenario-"+str(i) for i in [1, 2, 5, 6, 8, 9, 10, 11, 12, 14, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 39, 40, 42, 45, 46, 48, 49, 50, 51, 52, 53, 54, 56, 57]]
testNet1 = ["Scenario-"+str(i) for i in [70, 73, 76, 77, 81, 82, 83, 84, 3, 4]]

numClasses = 1

sequenceSize = 1
stepSize = 1
singleTargetValue = False  # None if the target and input sequence has the same size

normalizeInput = True
scenarios = []
lockDataSets = True

train = trainHanoi
test = testHanoi
numScenarios = 10  # used if no specific scenarios are given
percentTestScenarios = 1
network = "Hanoi_CMH"
# network = "Net1"
# network = "Net1_200scen_30days"
scenariosFolder = "NetworkModels/Benchmarks/" + network + "/"
inpFile = "NetworkModels/networks/" + network + ".inp"

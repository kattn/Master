loadModel = False
modelPath = "gru_hs30_nL2.pt"

# Small leaks [2, 5, 8, 8, 11, 13, 19, 20, 22, 32, 32, 34, 36, 38, 43, 51, 52, 69, 85, 96, 97, 104, 111, 120, 130, 132, 137, 140, 142, 147, 149, 152, 154, 156, 160, 166, 172, 172, 178, 181]
# med leaks [56, 84, 82, 35, 55, 106, 60, 80, 146, 72, 53, 186, 179, 113, 121, 20, 24, 33, 136, 29, 198, 12, 116, 170, 134, 6, 52, 136, 165, 26, 43, 95, 191, 189, 31, 117, 94, 71, 87, 173]
# big leaks [185, 63, 31, 21, 169, 142, 184, 162, 138, 112, 192, 51, 99, 55, 122, 91, 67, 130, 171, 79, 171, 193, 10, 67, 157, 16, 30, 144, 86, 177, 198, 155, 40, 167, 200, 123, 176, 152, 180, 127, 163, 40, 70, 149, 12, 49, 92, 197, 28, 99]
# big test 100, 109, 129, 76, 87, 63, 131, 128, 102

trainHanoi = ["Scenario-"+str(i) for i in [185, 63, 31, 21, 169, 142, 184, 162, 138, 112, 192, 51, 99, 55, 122, 91, 67, 130, 171, 79, 171, 193, 10, 67, 157, 16, 30, 144, 86, 177, 198, 155, 40, 167, 200, 123, 176, 152, 180, 127, 163, 40, 70, 149, 12, 49, 92, 197, 28, 99]]
testHanoi = ["Scenario-"+str(i) for i in [100, 109, 129, 76, 87, 63, 131, 128, 102]]

trainNet1 = ["Scenario-"+str(i) for i in [1, 2, 5, 6, 8, 9, 10, 11, 12, 14, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 39, 40, 42, 45, 46, 48, 49, 50, 51, 52, 53, 54, 56, 57]]
testNet1 = ["Scenario-"+str(i) for i in [70, 73, 76, 77, 81, 82, 83, 84, 3, 4]]

numClasses = 2

sequenceSize = 168
stepSize = 168
singleTargetValue = False  # None if the target and input sequence has the same size

normalizeInput = True
scenarios = []
lockDataSets = True

train = trainHanoi
test = testHanoi
numScenarios = 10  # used if no specific scenarios are given
percentTestScenarios = 0.5
network = "Hanoi_CMH"
# network = "Net1"
scenariosFolder = "NetworkModels/Benchmarks/" + network + "/"
inpFile = "NetworkModels/networks/" + network + ".inp"

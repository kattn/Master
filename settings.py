loadModel = False
modelPath = "models/singleGRU/singleGRU.pt"

numClasses = 1

sequenceSize = 72
stepSize = 72
singleTargetValue = False  # None if the target and input sequence has the same size

normalizeInput = True
scenarios = []
lockDataSets = True
train = ["Scenario-"+str(i) for i in range(1, 41)]
test = ["Scenario-"+str(i) for i in range(41, 51)]
numScenarios = 10  # used if no specific scenarios are given
percentTestScenarios = 0.5
network = "Hanoi_CMH"
# network = "Net1"
scenariosFolder = "NetworkModels/Benchmarks/" + network + "/"
inpFile = "NetworkModels/networks/" + network + ".inp"
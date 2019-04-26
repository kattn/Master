loadModel = False
modelPath = "singleGRU.1.pt"

numClasses = 1

sequenceSize = 1
stepSize = 1

normalizeInput = True
scenarios = []
numScenarios = 200  # used if no specific scenarios are given
percentTestScenarios = 0.25
network = "Net1"
scenariosFolder = "NetworkModels/Benchmarks/" + network + "/"
inpFile = "NetworkModels/networks/" + network + ".inp"
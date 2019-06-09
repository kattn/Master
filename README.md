# Master
My master thesis project, a prototype leak detection system using Gated RNNs. As it is a prototype, it was never meant to be easy to understand for others then me and as the scope of the thesis change, the old code remained, so beware that not all of the code here will be functioning.

# How to use it

## Generate data
Firstly, a data set must be generated. This is done through NetworkModels/leakDBGenerator.py. The generator uses the INP file set in the INP variable and simulates the network for the amount of days set in the num_days variable. The number of simulations done can be cset using the NumScenarios variable.

## Run the system
There are two ways to run the system, either by using a pretrained model or train a model. To run a pretrained model, set the settings of the model in the specific model file(either SingleLSTM or SingleGRU) and set the loadModel and modelPath variables in settings.py to True and the path to that specific model. Set the train and test variables to name which scenarios that should be used, and set the network variable to choose which networks data set is going to be used.

To train a model, set leadModel to False and set the wanted parameters in settings.py, main.py and the specific model file. A list of parameters and their functions will come soon, and also a clean up of the settings file(as it for now is full of my temporary variables).

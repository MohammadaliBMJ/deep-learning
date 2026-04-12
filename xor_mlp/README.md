# XOR Neural Network
This project implements how different numbers of hidden neurons affect the learning of the XOR function in a multilayer perceptron.
## Architecture
A simple feed-forward network with 2 layers. $\text{Input (2 features)} \rightarrow \text{Hidden (h neurons)} \rightarrow \text{Output (2 neurons)}$
The activation function of the hidden layer is Tanh function.
## Results
The models with 4 and 5 neurons in the hidden layer, learn the XOR function correctly and have the lowest errors as shown in the plot.

## How to run
Simply run the following command in your terminal.
```bash
python ./train.py
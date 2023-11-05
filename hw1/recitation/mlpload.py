import torch
import torch.nn as nn
import numpy as np
import pickle
import copy

class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.layer1 = nn.Linear(784, 64)
        self.layer2 = nn.Linear(64, 10)
        self.activation_function = nn.LeakyReLU()
        self.softmax_function = nn.Softmax(dim=1)

    def forward(self, x):
        hidden_layer_output = self.activation_function(self.layer1(x))
        output_layer = self.layer2(hidden_layer_output)
        return output_layer

x_test, y_test = pickle.load(open("data/mnist_test.data", "rb"))

x_test = x_test / 255.0
x_test = x_test.astype(np.float32)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).to(torch.long)
nn_model = MLPModel()
nn_model.load_state_dict(torch.load(open("mlp_model.mdl", "rb")))

softmax_function = torch.nn.Softmax(dim=1)
with torch.no_grad():
    predictions = nn_model(x_test)
    probability_scores = softmax_function(predictions)
    print(predictions)
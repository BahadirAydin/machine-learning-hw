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

    def forward(self, x):
        hidden_layer_output = self.activation_function(self.layer1(x))
        output_layer = self.layer2(hidden_layer_output)
        return output_layer

x_train, y_train = pickle.load(open("data/mnist_train.data", "rb"))
x_validation, y_validation = pickle.load(open("data/mnist_validation.data", "rb"))
x_test, y_test = pickle.load(open("data/mnist_test.data", "rb"))

x_train = x_train/255.0
x_train = x_train.astype(np.float32)

x_test = x_test / 255.0
x_test = x_test.astype(np.float32)

x_validation = x_validation/255.0
x_validation = x_validation.astype(np.float32)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).to(torch.long)

x_validation = torch.from_numpy(x_validation)
y_validation = torch.from_numpy(y_validation).to(torch.long)

x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).to(torch.long)

nn_model = MLPModel()
nn_model.train()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)

soft_max_function = torch.nn.Softmax(dim=1)

ITERATION = 250

for iteration in range(1, ITERATION +1):

    optimizer.zero_grad()
    predictions = nn_model(x_train)

    loss_value = loss_function(predictions, y_train)

    loss_value.backward()
    optimizer.step()

    with torch.no_grad():
        train_prediction = nn_model(x_train)
        train_loss = loss_function(train_prediction, y_train)
        predictions = nn_model(x_validation)
        probability_score_values = soft_max_function(predictions)
        validation_loss = loss_function(predictions, y_validation)
    print("Iteration : %d Training Loss : %f - Validation Loss %f" % (iteration, train_loss.item(), validation_loss.item()))


torch.save(nn_model.state_dict(), open("mlp_model.mdl", "wb"))

with torch.no_grad():
    predictions = nn_model(x_test)
    test_loss = loss_function(predictions, y_test)
    print("Test - Loss %.2f" % (test_loss))

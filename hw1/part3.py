import torch
import torch.nn as nn
import numpy as np
import pickle


# HYPERPARAMETERS IN THIS PART:
# 1. Number of hidden layers
# 2. Number of neurons in each hidden layer
# 3. Activation function
# 4. Learning rate
# 5. Number of epochs
# 6. Batch size (for mini-batch gradient descent)

# I am expected to test at least 10 hyperparameter configurations


class MLP(nn.Module):
    def __init__(
        self,
        hidden_layer_sizes,
        activation,
        input_size=784,
        output_size=10,
    ):
        super(MLP, self).__init__()
        if len(hidden_layer_sizes) == 0:
            raise ValueError("hidden_layer_sizes must have at least one element")
        self.input_layer = nn.Linear(input_size, hidden_layer_sizes[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_layer_sizes) - 1):
            self.hidden_layers.append(
                nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
            )
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], output_size)
        self.activation = activation

    def forward(self, input):
        output = self.activation(self.input_layer(input))
        for layer in self.hidden_layers:
            output = self.activation(layer(output))
        return self.output_layer(output)


class Trainer:
    def __init__(
        self,
        train,
        validation,
        test,
        epochs,
        batch_sizes,
        learning_rates,
        activation_functions,
        hidden_layer_sizes,
    ):
        # Model is the neural network model (class MLP)
        self.train_data = train
        self.validation_data = validation
        self.test_data = test
        self.epochs = epochs
        self.batch_sizes = batch_sizes
        self.learning_rates = learning_rates
        self.activation_functions = activation_functions
        self.hidden_layer_sizes = hidden_layer_sizes

        self.results = []

    def train(
        self, epoch, batch_size, learning_rate, activation_function, hidden_layer_sizes
    ):
        network = MLP(hidden_layer_sizes, activation_function)
        optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
        loss = nn.CrossEntropyLoss()
        # set the network to training mode
        network.train()

        for _ in range(epoch):
            x_train, y_train = self.train_data
            for j in range(0, len(x_train), batch_size):
                x_batch = x_train[j : j + batch_size]
                y_batch = y_train[j : j + batch_size]
                # to reset the gradients of all parameters
                optimizer.zero_grad()
                output = network(x_batch)
                loss_value = loss(output, y_batch)
                loss_value.backward()
                optimizer.step()
        return network

    def test(self, network):
        # set the network to evaluation mode
        network.eval()
        x_test, _ = self.test_data
        with torch.no_grad():
            predictions = network(x_test)
        return predictions

    def validate(self, network):
        # set the network to evaluation mode
        network.eval()
        x_validation, y_validation = self.validation_data
        with torch.no_grad():
            predictions = network(x_validation)
        return predictions

    def calculate_accuracy(self, predictions, labels):
        with torch.no_grad():
            return (
                torch.mean((torch.argmax(predictions, dim=1) == labels).float()) * 100
            )

    def run(self):
        for epoch in self.epochs:
            for batch_size in self.batch_sizes:
                for learning_rate in self.learning_rates:
                    for activation_function in self.activation_functions:
                        for hidden_layer_size in self.hidden_layer_sizes:
                            test_mean = 0
                            validation_mean = 0
                            print(
                                "Training the network with hyperparameters:\nEpoch: {}\nBatch Size: {}\nLearning Rate: {}\nActivation Function: {}\nHidden Layer Size: {}".format(
                                    epoch,
                                    batch_size,
                                    learning_rate,
                                    activation_function,
                                    hidden_layer_size,
                                )
                            )
                            for i in range(10):
                                network = self.train(
                                    epoch,
                                    batch_size,
                                    learning_rate,
                                    activation_function,
                                    hidden_layer_size,
                                )
                                predictions = self.test(network)
                                test_accuracy = self.calculate_accuracy(
                                    predictions, y_test
                                )
                                test_mean += test_accuracy
                                print(
                                    "Iteration {} - Accuracy: {:2f}".format(
                                        i, test_accuracy.item()
                                    )
                                )
                                validation_predictions = self.validate(network)
                                validation_accuracy = self.calculate_accuracy(
                                    validation_predictions, y_validation
                                )
                                validation_mean += validation_accuracy
                            test_mean = test_mean / 10
                            validation_mean = validation_mean / 10
                            print(
                                "Test mean accuracy of 10 iterations: {:2f}".format(
                                    test_mean
                                )
                            )
                            self.results.append(
                                (
                                    epoch,
                                    batch_size,
                                    learning_rate,
                                    activation_function,
                                    hidden_layer_size,
                                    test_mean,
                                    validation_mean,
                                )
                            )

        filename = "results.txt"
        with open(filename, "w") as f:
            for result in self.results:
                f.write(
                    "Epoch: {}\nBatch Size: {}\nLearning Rate: {}\nActivation Function: {}\nHidden Layer Size: {}\nMean Accuracy: {}\n".format(
                        result[0],
                        result[1],
                        result[2],
                        result[3],
                        result[4],
                        result[5],
                    )
                )


x_train, y_train = pickle.load(open("data/mnist_train.data", "rb"))
x_validation, y_validation = pickle.load(open("data/mnist_validation.data", "rb"))
x_test, y_test = pickle.load(open("data/mnist_test.data", "rb"))

x_train = x_train / 255.0
x_train = x_train.astype(np.float32)

x_test = x_test / 255.0
x_test = x_test.astype(np.float32)

x_validation = x_validation / 255.0
x_validation = x_validation.astype(np.float32)

# and converting them into Pytorch tensors in order to be able to work with Pytorch
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).to(torch.long)

x_validation = torch.from_numpy(x_validation)
y_validation = torch.from_numpy(y_validation).to(torch.long)

x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).to(torch.long)


hidden_layer_sizes = [(16,), (32,), (64,), (16, 16), (16, 32)]
activation_functions = [nn.ReLU(), nn.Sigmoid()]  # nn.Tanh()
learning_rates = [0.1, 0.01, 0.001]
num_epochs = [10, 20, 40, 50]
batch_sizes = [16, 32, 64]
# Total there are 6 * 2 * 3 * 5 * 4 = 720 hyperparameter configurations

trainer = Trainer(
    (x_train, y_train),
    (x_validation, y_validation),
    (x_test, y_test),
    num_epochs,
    batch_sizes,
    learning_rates,
    activation_functions,
    hidden_layer_sizes,
)
trainer.run()

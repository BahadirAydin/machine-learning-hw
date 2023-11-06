import torch
import numpy as np
import pickle
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def softmax(vector):
    e = torch.exp(vector)
    return e / e.sum()


# this function returns the neural network output for a given dataset and set of parameters
def forward_pass(w1, b1, w2, b2, input_data):
    """
    The network consists of 3 inputs, 16 hidden units, and 3 output units
    The activation function of the hidden layer is sigmoid.
    The output layer should apply the softmax function to obtain posterior probability distribution. And the function should return this distribution
    Here you are expected to perform all the required operations for a forward pass over the network with the given dataset
    """
    predictions = []
    for row in input_data:
        r = row.reshape(3, 1)
        z_1 = torch.matmul(w1, r) + b1
        a_1 = sigmoid(z_1)

        z2 = torch.matmul(w2, a_1) + b2
        a2 = softmax(z2)
        predictions.append(a2.flatten())
    return torch.stack(predictions)


def mean_crossentropy_loss(predictions, labels):
    return -torch.mean(labels * torch.log(predictions))


def calculate_accuracy(predictions, labels):
    # axis 0 means column wise
    # we will look at each prediction and label pair and check if the index of the maximum value in the prediction vector is equal to the index of the maximum value in the label vector
    # numpy creates a boolean array with the same size as the prediction and label vectors
    # if the prediction and label pair is equal, the corresponding index in the boolean array will be True, otherwise False
    accuracy_list = torch.argmax(predictions,dim=1) == torch.argmax(labels, dim=1)
    accuracy = torch.mean(accuracy_list.float())*100
    return accuracy


# LOAD DATASETS
train_dataset, train_label = pickle.load(
    open("data/part2_classification_train.data", "rb")
)
validation_dataset, validation_label = pickle.load(
    open("data/part2_classification_validation.data", "rb")
)
test_dataset, test_label = pickle.load(
    open("data/part2_classification_test.data", "rb")
)

# when you inspect the training dataset, you are going to see that the class instances are sequential (e.g [1,1,1,1 ... 2,2,2,2,2 ... 3,3,3,3])
# we shuffle the training dataset by preserving instance-label relationship
indices = list(range(len(train_dataset)))
np.random.shuffle(indices)
train_dataset = np.array([train_dataset[i] for i in indices], dtype=np.float32)
train_label = np.array([train_label[i] for i in indices], dtype=np.float32)

# CONVERT TO TENSOR
train_dataset = torch.from_numpy(train_dataset)
train_label = torch.from_numpy(train_label)

validation_dataset = torch.from_numpy(validation_dataset)
validation_label = torch.from_numpy(validation_label)

test_dataset = torch.from_numpy(test_dataset)
test_label = torch.from_numpy(test_label)

# You are expected to create and initialize the parameters of the network
# Please do not forget to specify requires_grad=True for all parameters since they need to be trainable.

# w1 defines the parameters between the input layer and the hidden layer
w1 = torch.zeros(16, 3, requires_grad=True)

mean_w1, std_w1 = 0, 1
with torch.no_grad():
    w1.normal_(mean_w1, std_w1)

# b defines the bias parameters for the hidden layer
b1 = torch.zeros(16, 1, requires_grad=True)

mean_b1, std_b1 = 0, 1
with torch.no_grad():
    b1.normal_(mean_b1, std_b1)

# w2 defines the parameters between the hidden layer and the output layer
w2 = torch.zeros(3, 16, requires_grad=True)

mean_w2, std_w2 = 0, 1
with torch.no_grad():
    w2.normal_(mean_w2, std_w2)

# b2 defines the bias parameters for the output layer
b2 = torch.zeros(3, 1, requires_grad=True)
mean_b2, std_b2 = 0, 1
with torch.no_grad():
    b2.normal_(mean_b2, std_b2)


# Stochastic gradient descent optimizer
# Trainable parameters: w1, b1, w2, b2
optimizer = torch.optim.SGD([w1, b1, w2, b2], lr=0.001)

# These arrays will store the loss values incurred at every training iteration
iteration_array = []
train_loss_array = []
validation_loss_array = []

# We are going to perform the backpropagation algorithm 'ITERATION' times over the training dataset
# After each pass, we are calculating the cross entropy loss over the validation dataset along with accuracy scores on both datasets.
ITERATION = 15000
for iteration in range(1, ITERATION + 1):
    iteration_array.append(iteration)

    # we need to zero all the stored gradient values calculated from the previous backpropagation step.
    optimizer.zero_grad()
    # Using the forward_pass function, we are performing a forward pass over the network with the training data
    train_predictions = forward_pass(w1, b1, w2, b2, train_dataset)
    # here you are expected to calculate the MEAN cross-entropy loss with respect to the network predictions and the training label
    train_mean_crossentropy_loss = mean_crossentropy_loss(
        train_predictions, train_label
    )

    train_loss_array.append(train_mean_crossentropy_loss.item())

    # We initiate the gradient calculation procedure to get gradient values with respect to the calculated loss
    train_mean_crossentropy_loss.backward()
    # After the gradient calculation, we update the neural network parameters with the calculated gradients.
    optimizer.step()

    # after each epoch on the training data we are calculating the loss and accuracy scores on the validation dataset
    with torch.no_grad():
        # Here you are expected to calculate the accuracy score on the training dataset by using the training labels.
        train_accuracy = calculate_accuracy(train_predictions, train_label)

        validation_predictions = forward_pass(w1, b1, w2, b2, validation_dataset)

        # Here you are expected to calculate the average/mean cross entropy loss for the validation datasets by using the validation dataset labels.
        validation_mean_crossentropy_loss = mean_crossentropy_loss(
            validation_predictions, validation_label
        )

        validation_loss_array.append(validation_mean_crossentropy_loss.item())

        validation_accuracy = calculate_accuracy(
            validation_predictions, validation_label
        )

    print(
        "Iteration : %d - Train Loss %.4f - Train Accuracy : %.2f - Validation Loss : %.4f Validation Accuracy : %.2f"
        % (
            iteration + 1,
            train_mean_crossentropy_loss.item(),
            train_accuracy,
            validation_mean_crossentropy_loss.item(),
            validation_accuracy,
        )
    )


# after completing the training, we calculate our network's accuracy score on the test dataset...
with torch.no_grad():
    test_predictions = forward_pass(w1, b1, w2, b2, test_dataset)
    test_accuracy = calculate_accuracy(test_predictions, test_label)
    print("Test accuracy : %.2f" % (test_accuracy.item()))

# We plot the loss versus iteration graph for both datasets (training and validation)
plt.plot(iteration_array, train_loss_array, label="Train Loss")
plt.plot(iteration_array, validation_loss_array, label="Validation Loss")
plt.legend()
plt.show()

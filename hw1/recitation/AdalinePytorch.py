import numpy as np
import math
import matplotlib.pyplot as plt
from boundary_display import display_boundary
import time
import torch

# each data instance (sample) is represented with a row vector
# here the dataset consists of , and each data sample has 2 features
input_data = torch.FloatTensor([[0.4, 0.4], # first data instance [0.4, 0.4]
                       [-0.4, 0.4], # second data instance [-0.4, 0.4]
                       [0.4, 0.6],
                       [-0.3,0.2],
                       [0, 0.7],
                       [-0.5, -0.6]])
input_data_label = torch.LongTensor([[1], [-1], [1], [-1], [1], [-1]])


nonlinearly_separable_inpur_data = torch.FloatTensor([[0.2, 0.1],
                       [-0.1, 0.6],
                       [-0.2, 0.6],
                       [-0.5,0.4],
                       [0.3, 0.7],
                       [-0.1, -0.8]])

nonlinearly_separable_input_data_label = torch.LongTensor([[1], [-1], [1], [-1], [1], [-1]])

# We initialize the parameters of the adaline model randomly (note: adaline model differs in training procedure from the perceptron model
# normal vector of the hyperplane
# with mean 0 and std=1 (normal distribution),
# we generate an array (column vector, or 2x1 matrix) with 2 elements
# W = |w1|
#     |w2|
W = torch.from_numpy(np.random.normal(0, 1, 2).astype(np.float32).reshape((2, 1))).requires_grad_(True) # with mean 0 and std=1 (normal distribution),
# bias of the hyperplane, w0
# we generate an array (1-by-1 matrix) with 1 element
# b=|w0|
b = torch.from_numpy(np.random.normal(0, 1, 1 ).astype(np.float32).reshape((1,1))).requires_grad_(True) # the same initialization for the bias of the perceptron (b)

# initial boundary after weight initialization
with torch.no_grad():
    display_boundary(input_data, W, b, input_data_label)

ITERATION = 150
lr = 0.01

optimizer = torch.optim.SGD([W, b], lr=lr)

for iteration in range(1, ITERATION+1):
    optimizer.zero_grad()
    dot_product = torch.matmul(input_data, W)
    output = dot_product + b
    predicted_label = torch.sign(output)

    diff = input_data_label-output
    loss_value = torch.sum(diff**2)/2
    loss_value.backward()
    optimizer.step()
    with torch.no_grad():
        display_boundary(input_data, W, b, input_data_label)
print("Final W:", (W[0][0], W[1][0]))
print("Final b:", b[0][0])


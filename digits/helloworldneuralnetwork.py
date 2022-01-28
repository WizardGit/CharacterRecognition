import torch
import torch.nn as nn
import torch.optim as optim

"""
torch contains all PyTorch utilities
torch.nn is used for neural network operations and torch.optim is for neural network optimizers
1. build a computation graph
2. set up optimizers
3. set up criterion
4. set up data
5. train the model
"""

# now we define the neural network, training utilities, and the dataset:
net = nn.Linear(1,1) # we are just creating a straight line here (computation graph)
# using stochastic gradient descent
optimizer = optim.SGD(net.parameters(), lr=0.1) # setting up the optimizers: here we are specifying the step size! (think of the bowl)
criterion = nn.MSELoss() # setting up the criterion
x, target = torch.randn((1,)), torch.tensor([0.]) # setting up the data we'll use

# the model is a line of the form y = mx, and the parameter nn.Linear(1,1) is the slop of the line
# this model parameter (the slope) will get updated during training

# the optimizer will determine how the neural network will learn

# the criterion defines the loss, which means what the model is trying to minimize
# we want minimize the difference between the line's predicted y-values and the actual y-values in the training set.

# x, target defines our dataset, which for right now is just a coordinate (and x and a y value)

# we'll train our model by iterating over the dataset a fixed amount of times (10 for now)
# each time we'll adjust the model's parameter


# TRAIN THE MODEL (using gradient descent)
iterator = 10
# equivalent to stepping closer to the center of the bowl!
# the negative gradient points to the lowest point in the bowl
for k in range(iterator):
    output = net(x)
    loss = criterion(output, target)
    print(round(loss.item(), 2))

    net.zero_grad() # have to clear the gradient or else it will keep getting added up
    loss.backward() # compute new gradients
    optimizer.step() # uses the gradients to take steps

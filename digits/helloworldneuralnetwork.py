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
network = nn.Linear(1,1) # we are just creating a straight line here (computation graph)
optimizer = optim.SGD(network.parameters(), lr=0.1) # setting up the optimizers
criterion = nn.MSELoss() # setting up the criterion
x, target = torch.randn((1,)), torch.tensor([0.]) # setting up the data we'll use
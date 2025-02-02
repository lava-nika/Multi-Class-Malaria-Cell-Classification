import torch
import torch.nn as nn
import torch.nn.functional as F

class MalariaMLP(nn.Module):
    def __init__(self, input_size=30000, hidden_sizes=[256, 128], output_size=6):
        super(MalariaMLP, self).__init__()
        # Flatten the input layer
        self.flatten = nn.Flatten()
        
        # Hidden layers
        self.hidden1 = nn.Linear(input_size, hidden_sizes[0])
        self.hidden2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        
        # Output layer
        self.output = nn.Linear(hidden_sizes[1], output_size)
    
    def forward(self, x):
        # Flatten the input
        x = self.flatten(x)
        
        # Pass through hidden layers with ReLU activation
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        
        # Pass through the output layer
        x = self.output(x)
        
        # Apply softmax to output probabilities
        return F.log_softmax(x, dim=1)

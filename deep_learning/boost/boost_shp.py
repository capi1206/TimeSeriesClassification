import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from deep_learning.boost import BoostingRNN 
from deep_learning.utils import prepare_data_random, shuffle_data


# Set seed for reproducibility
torch.manual_seed(0)

# Hyperparameters
input_dim = 145  # Number of features per time step
hidden_dim = 16
output_dim = 1  # Binary classification
sequence_length = 10
batch_size = 8
num_models = 2
num_epochs = 2
learning_rate = 0.01

data = np.load("few_series/np_series_concat2.npy")
# cols = [0, 1, 3, 5, 13, 15, 144, 16, 17, 19, 20, 21, 23, 24, 27, 28, 29, 30, 31, 34, 36, 38, 39, 40, 41, 46, 48, 49, 52, 53, 54, 55, 56, 57, 59, 61, 62, 63, 64, 67, 68, 71, -1]
# data = data[:,:, cols]
prepared_data = prepare_data_random(data)
data = (
    shuffle_data(prepared_data[0][0][:32], prepared_data[0][1][:32]),
    shuffle_data(prepared_data[1][0], prepared_data[1][1]),
    shuffle_data(prepared_data[2][0], prepared_data[2][1]),
)
train_dataset = TensorDataset(*data[0])
#val_dataset = TensorDataset(*data[1])
#test_dataset = TensorDataset(*data[2])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=hyperparams["batch_size"])
#test_loader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"])
# Create DataLoader
#dataset = TensorDataset(x_data[:32,], y_data.float())
#train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize and train the BoostingRNN model
boosting_rnn = BoostingRNN(input_dim, hidden_dim, output_dim, num_models=num_models, device='cpu', lr=learning_rate, details=True)
boosting_rnn.fit(train_loader, num_epochs)

# Test prediction on a single sample
test_sample = torch.randn(1, sequence_length, input_dim)
predicted_output = boosting_rnn.predict(test_sample)
print(f"Predicted output: {predicted_output}")
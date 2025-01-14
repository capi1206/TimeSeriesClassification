##BoostRNN
import torch
import torch.nn as nn
import torch.optim as optim

#Base RNN Model
class BaseRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, details=False):
        super(BaseRNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.details = details
    
    def forward(self, x):
        if self.details:
            print(f" base rnn x s :{x.shape}")
        _, (h_n, _) = self.lstm(x) 
        output = self.fc(h_n[-1])  # Output is of shape (batch_size, 1) for binary classification
        if self.details:
            print(f" output s: {output.shape}")
        return output

# Boosting BoostingRNNModel
class BoostingRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_models=5, 
                 num_layers=1, 
                 device='cpu', 
                 lr=0.01, 
                 details=False):
        super(BoostingRNN, self).__init__()
        self.models = [
            BaseRNN(input_dim, hidden_dim, output_dim, num_layers, details).to(device)
            for _ in range(num_models)
        ]
        self.details = details
        self.device = device
        self.lr = lr
        self.num_models = num_models
        self.criterion = nn.BCEWithLogitsLoss() # For binary classification use BCEWithLogitsLoss
    
    def fit(self, train_loader, num_epochs):
        """
        Function defined to train the model rename to prevent confusion with torch.train.
        Configured to make the remaining chunk to fit the batch size since it feeds recurrently
        shapes need to match. 
    
        """
        residuals = None
        for model_idx, model in enumerate(self.models):
            #print(f" doing model {model_idx} out of {len(self.models)}")

            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            
            for epoch in range(num_epochs):
                
                model.train()
                
                for x_batch, y_batch in train_loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    if model.details:
                        print(f" x_batch shape: {x_batch.shape}, y_batch: {y_batch.shape}")
                    
                    # Compute model output
                    outputs = model(x_batch)
                    if model.details:
                        print(f" output shape {outputs.shape}, y_batch shape: {y_batch.shape}")
                    
                    if residuals is not None:
                    # Match residuals to the batch size, residuals were producing shape missmatch 
                        y_batch = residuals[:x_batch.size(0)]
                        if model.details:
                            print(f"Adjusted y_batch shape to residuals: {y_batch.shape}")

                    # Compute the loss
                    loss = self.criterion(outputs, y_batch.unsqueeze(-1))
                    if model.details:
                        print("Loss computed")

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            #print(f"Model {model_idx + 1}, Epoch {epoch + 1}, Loss: {loss.item():.4f}")


        residuals = y_batch - outputs.detach()
    
    def predict(self, x):
        predictions = torch.zeros(x.size(0), 1).to(self.device)
        for model in self.models:
            predictions += model(x)
        return predictions / self.num_models  # Average predictions
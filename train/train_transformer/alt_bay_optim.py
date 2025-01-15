## This one is an alternative approach to bayesian optimization training in samples ranomly from each coin
import copy  
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna


from models.transformer import Transformer
from models.utils import prepare_data_random, shuffle_data

seq_length = 30
test_perc = 0.05
val_perc = 0.1

def objective(trial, data, file_path):
    hyperparams = {
        "d_model": trial.suggest_categorical("d_model", [64, 120, 256]),
        "n_head": trial.suggest_categorical("n_head", [2, 4, 8]),
        "ffn_hidden": trial.suggest_categorical("ffn_hidden", [80, 160, 320]),
        "n_layers": trial.suggest_categorical("n_layers", [2, 4]),
        "drop_prob": trial.suggest_uniform("drop_prob", 0.2, 0.4),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "l1_lambda": trial.suggest_loguniform("l1_lambda", 1e-5, 1e-2),
    }
    val_accuracy = train_and_evaluate_model(hyperparams, data, file_path)
    return val_accuracy

def train_and_evaluate_model(hyperparams, data, file_path):
    """Trains and evaluates the model, function used in the bayesian optimization
    Args:
        hyperparams the hyperparams to perform the search on.
        the input data was loaded separated and shuffled before. 
        file_path "*.log" string where the log of the training will be saved.
    Returns:
        the objective for the bayesian search to optimize.    
    """
    best_model_params = None
    input_dim = data[0][0].shape[2]  # Number of features
    epochs = 150
    patience = 30
    device = 'cpu' #change if cuda is available
    model = Transformer(
        device=device,
        d_model=hyperparams["d_model"],
        n_head=hyperparams["n_head"],
        input_dim=input_dim,
        seq_len=seq_length,
        ffn_hidden=hyperparams["ffn_hidden"],
        n_layers=hyperparams["n_layers"],
        drop_prob=hyperparams["drop_prob"],
        l1_lambda=hyperparams["l1_lambda"]
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
   
    
    X_tr_t, y_tr_t = data[0]
    X_val_t, y_val_t = data[1]
    X_tst_t, y_tst_t = data[2]
    lenn = y_tr_t.size(0)
    n_pos = torch.sum(y_tr_t).item()
    pos_weight = (lenn - n_pos) / n_pos

    with open(file_path, "w") as f:
        best_v_acc = 0
        n_noimprov = 0
        correct = 0
        total = 0
        print(f"Training ...")


        train_dataset = TensorDataset(X_tr_t, y_tr_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'])
        # Train on this coin for one epoch
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

        for e in range(epochs):
            train_loss = 0.0
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
                y_batch = y_batch.float()

            # Forward pass
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                l1_loss = model.l1_loss()
                total_loss = loss + l1_loss

            # Backward pass and optimization
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                train_loss += total_loss.item() 
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).float()
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

            train_loss /= total
            train_accuracy =  100* correct / total

            val_loss = 0.0
            correct = 0
            total = 0
            model.eval()
            with torch.no_grad():
                val_dataset = TensorDataset(X_val_t, y_val_t)
                val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'])

                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)

                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)

                    val_loss += loss.item() * X_batch.size(0)
                    probabilities = torch.sigmoid(outputs)
                    predicted = (probabilities > 0.5).float()
                    correct += (predicted == y_batch).sum().item()
                    total += y_batch.size(0)

            val_loss /= total
            val_accuracy = 100 * correct / total

        # Early stopping and saving the best model
            if val_accuracy > best_v_acc:
                n_noimprov = 0
                best_v_acc = val_accuracy
                best_model_params = copy.deepcopy(model.state_dict())
                f.write(f" Best acc changed to {best_v_acc}, at epoch {e}/{epochs}\n")
                print(f" Best acc changed to {best_v_acc}, at epoch {e}/{epochs}\n")
                
            else:
                n_noimprov += 1

            if best_v_acc > 50 and n_noimprov > patience:
                f.write(f"Early stopping triggered, with best val acc: {best_v_acc}, rd")
                model.load_state_dict(best_model_params)
                break
            # if best_v_acc > 57 and n_noimprov > patience_2:
            #     f.write(f"Early stopping triggered, with best val acc: {best_v_acc}, nd")
            #     model.load_state_dict(best_model_params)
            #     break
            # if n_noimprov > patience_3:
            #     f.write(f"Early stopping triggered, with best val acc: {best_v_acc}, st")
            #     model.load_state_dict(best_model_params)
            #     break
                
            if e % 10 == 0:
                f.write(f"Epoch {e + 1}/{epochs}, \n"
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, \n"
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}% \n")
            print(f"Epoch {e + 1}/{epochs}, \n"
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, \n"
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}% \n")
                    
            if e ==epochs -1 and best_model_params:
                model.load_state_dict(best_model_params)
    
    
    return best_v_acc

file_path = "few_series/alt_bayesian_optim.log"  




data = np.load("few_series/np_series_concat2.npy")
cols = [0, 1, 3, 5, 13, 15, 144, 16, 17, 19, 20, 21, 23, 24, 27, 28, 29, 30, 31, 34, 36, 38, 39,40, 41, 46, 48, 49, 52, 53, 54, 55, 56, 57, 59, 61, 62, 63, 64, 67, 68, 71, -1]
data=data[:,:, cols]
data_res = prepare_data_random(data) 
data = (shuffle_data(data_res[0][0], data_res[0][1]), 
                     shuffle_data(data_res[1][0], data_res[1][1]), 
                     shuffle_data(data_res[2][0], data_res[2][1]))
print("shuffed data 0 shape: ", data[0][0].shape)

study = optuna.create_study(direction="maximize")
study.optimize(partial(objective, data=data, file_path=file_path), n_trials=50)

with open(file_path, 'a') as f:
    f.write("Best hyperparameters:", study.best_params)
    f.write("Best validation accuracy:", study.best_value)
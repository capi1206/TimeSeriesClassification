import os
import sys
import itertools
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import optuna
from functools import partial

from models.transformer import Transformer
from models.utils import norm_mts_with_window, create_seq_from_ts

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
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "l1_lambda": trial.suggest_loguniform("l1_lambda", 1e-5, 1e-2),
    }
    val_accuracy = train_and_evaluate_model(hyperparams, data, file_path)
    return val_accuracy

def train_and_evaluate_model(hyperparams, coin_data, file_path):
    """Trains and evaluates the model, function used in the bayesian optimization
    Args:
        hyperparams the hyperparams to oerform the search on.
        coin_data dictionary containing the different coins to evaluate.
        file_path "*.log" string where the log of the training will be saved.
    Returns:
        the objective for the bayesian search to optimize.    
    """
    best_model_params = None
    coin_ids = list(coin_data.keys())  # List of coin IDs
    input_dim = 145
    inner_epochs = 150
    patience_1 = 30
    patience_2 = 20
    patience_3 = 10
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
    best_v_acc_total = 0
    train_loss = 0.0
    correct = 0
    total = 0

    with open(file_path, "w") as f:
        for coin_id in coin_ids:
            best_v_acc = 0
            n_noimprov = 0
            print(f"Training coin {coin_id}...")
            coin_d = coin_data[coin_id]
            X_tr_t = torch.tensor(coin_d['X_tr'], dtype=torch.float32).to(device)
            y_tr_t = torch.tensor(coin_d['y_tr'], dtype=torch.float32).to(device)
            X_val_t = torch.tensor(coin_d['X_val'], dtype=torch.long).to(device)
            y_val_t = torch.tensor(coin_d['y_val'], dtype=torch.long).to(device)

            train_dataset = TensorDataset(X_tr_t, y_tr_t)
            val_dataset = TensorDataset(X_val_t, y_val_t)
            train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'])
            # Train on this coin for one epoch
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(coin_d["pos_weight"]))

            for e in range(inner_epochs):
                
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
                    for coin_id in coin_ids:
                        coin_d = coin_data[coin_id]
                        X_val_t = torch.tensor(coin_d['X_val'], dtype=torch.float32).to(device)
                        y_val_t = torch.tensor(coin_d['y_val'], dtype=torch.float32).to(device)

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
                    if val_accuracy > best_v_acc_total:
                        best_v_acc_total = val_accuracy
                        best_model_params = copy.deepcopy(model.state_dict())
                        f.write(f" Total best acc changed to {best_v_acc_total}, at epoch {e}/{inner_epochs}\n")
                    
                else:
                    n_noimprov += 1

                if best_v_acc > 70 and n_noimprov > patience_1:
                    f.write(f"Early stopping triggered, with best val acc: {best_v_acc}, rd")
                    model.load_state_dict(best_model_params)
                    break
                if best_v_acc > 57 and n_noimprov > patience_2:
                    f.write(f"Early stopping triggered, with best val acc: {best_v_acc}, nd")
                    model.load_state_dict(best_model_params)
                    break
                if n_noimprov > patience_3:
                    f.write(f"Early stopping triggered, with best val acc: {best_v_acc}, st")
                    model.load_state_dict(best_model_params)
                    break
                    
                if e % 10 == 0:
                    f.write(f"Epoch {e + 1}/{inner_epochs}, \n"
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, \n"
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}% \n")
                    
                if e ==inner_epochs -1 and best_model_params:
                    model.load_state_dict(best_model_params)
    
    
    return best_v_acc_total

file_path = "few_series/bayesian_optim_alternative.log"        

def prepare_data(data):
    '''This function put the data into three different arrays,
    respectively train, validation and test. Keeping the Easliest data for testing and 
    the further in time data for training.
    retuns: (data_training(x,y), data_validation(x,y), data_testing(x,y))
    '''
    data_tr_x, data_val_x, data_tst_x = [], [], [] 
    data_tr_y, data_val_y, data_tst_y = [], [], []
    
    for j in range(data.shape[0]):  
        xdata = data[j, -1500:,:] #take only the last 1500 measurements
        data_2_norm = xdata[:, :-1]

        # Check for NaNs
        if np.isnan(data_2_norm).any().any():
            print(f"Something wrong at {j}")
            continue
        
        data_2_norm = norm_mts_with_window(data_2_norm)

        # Create sequences
        sequences = create_seq_from_ts(data_2_norm, seq_length)
        if np.isnan(sequences).any().any():
            print(f"Something wrong at sequences in {j}")
            continue

        # Prepare targets
        y = xdata[seq_length - 1:, -1]

        # Split into training, validation, and testing
        val_i = int(sequences.shape[0] * val_perc)
        test_i = int(sequences.shape[0] * test_perc)
        X_tr, X_val, X_tst = sequences[:-(val_i + test_i), :, :], sequences[-(val_i + test_i):-test_i, :, :], sequences[-test_i:, :, :]
        y_tr, y_val, y_tst = y[:-(val_i + test_i)], y[-(val_i + test_i):-test_i], y[-test_i:]
        y_val = (y_val + 1) / 2 
        y_tr = (y_tr + 1) / 2
        y_tst = (y_tst + 1) / 2  
        n_pos_tr = y_tr.sum()
        weight = (len(y_tr) -n_pos_tr)/n_pos_tr
        data_tr_x.append(X_tr)
        data_val_x.append(X_val)
        data_tst_x.append(X_tst)
        data_tr_y.append(y_tr)
        data_val_y.append(y_val)
        data_tst_y.append(y_tst)
    data_tr_x = np.concatenate(data_tr_x, axis=0)
    data_val_x = np.concatenate(data_val_x, axis=0) 
    data_tst_x = np.concatenate(data_tst_x, axis=0) 
    data_tr_y = np.concatenate(data_tr_y, axis=0)
    data_val_y = np.concatenate(data_val_y, axis=0) 
    data_tst_y = np.concatenate(data_tst_y, axis=0)
    print(" data tr x shape :", data_tr_x.shape)
    print(" data tr y shape :", data_tr_y.shape)    
    return ((data_tr_x, data_tr_y), 
            (data_val_x, data_val_y),
            (data_tst_x, data_tst_y)) 

data = np.load("few_series/np_series_concat2.npy")
coin_data = prepare_data(data)     

study = optuna.create_study(direction="maximize")
study.optimize(partial(objective, data=coin_data, file_path=file_path), n_trials=50)

with open(file_path, 'a') as f:
    f.write("Best hyperparameters:", study.best_params)
    f.write("Best validation accuracy:", study.best_value)
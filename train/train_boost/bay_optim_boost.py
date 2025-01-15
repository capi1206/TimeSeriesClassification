import copy  
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna

from models.boost import BoostingRNN 
from models.utils import prepare_data_random, shuffle_data

seq_length = 30
test_perc = 0.05
val_perc = 0.1

BEST_MODEL_FILE = "few_series/train_boost/model_results/best_model_params.pth"
best_val_acc_over_trials = 0

def objective(trial, data):
    hyperparams = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
        "num_layers": trial.suggest_categorical("num_layers", [1, 2, 3]),
        "num_models": trial.suggest_categorical("num_models", [2,4, 6]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
    }

    val_accuracy = train_and_evaluate_boosting_rnn(hyperparams, data)
    return val_accuracy

def train_and_evaluate_boosting_rnn(hyperparams, data):

    global best_val_acc_over_trials
    input_dim = data[0][0].shape[2]  # Number of features
    output_dim = 1  # Binary classification
    device = 'cpu' #change if gpu available
    epochs = 50
    patience = 10

    boosting_rnn = BoostingRNN(
        input_dim=input_dim,
        hidden_dim=hyperparams["hidden_dim"],
        output_dim=output_dim,
        num_models=hyperparams["num_models"],
        num_layers=hyperparams["num_layers"],
        device=device,
        lr=hyperparams["lr"],
    )

    train_dataset = TensorDataset(*data[0])
    val_dataset = TensorDataset(*data[1])
    test_dataset = TensorDataset(*data[2])
    train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"])

    best_val_accuracy = 0
    no_improve_count = 0

    for epoch in range(epochs):
    
        boosting_rnn.fit(train_loader, num_epochs=1)  # Train for one epoch

        # Evaluate on validation data
        train_accuracy = evaluate(boosting_rnn, train_loader, device)
        val_accuracy = evaluate(boosting_rnn, val_loader, device)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f" Best acc changed to {best_val_accuracy:.2f}%, at epoch {epoch}/{epochs}\n")
            
            tst_corresp_acc = evaluate(boosting_rnn, test_loader, device)
            print(f" with corresp tst accuracy: {tst_corresp_acc:.2f}%")
            if val_accuracy > best_val_acc_over_trials :
                best_val_acc_over_trials = val_accuracy
                best_model_params = copy.deepcopy(boosting_rnn.state_dict())
                torch.save(best_model_params, BEST_MODEL_FILE)
                print(f"Model saved to {BEST_MODEL_FILE}")
            no_improve_count = 0
        else:
            no_improve_count += 1
            if epoch % 10 == 0:
                print(f"At epoch {epoch}/{epochs} we have {train_accuracy} % train acc,")
                print(f"and {val_accuracy}% val accuracy ")
        if no_improve_count > patience:
            print("Early stopping triggered")
            break        

    return best_val_accuracy

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device).unsqueeze(1)
            if model.details:
                print("before predict")
            outputs = model.predict(x_batch)
            if model.details: print("did preds!!")
            predictions = (outputs > 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
                     

    return correct / total

data = np.load("few_series/np_series_concat2.npy")

prepared_data = prepare_data_random(data)
data = (
    shuffle_data(prepared_data[0][0], prepared_data[0][1]),
    shuffle_data(prepared_data[1][0], prepared_data[1][1]),
    shuffle_data(prepared_data[2][0], prepared_data[2][1]),
)

study = optuna.create_study(direction="maximize")
study.optimize(partial(objective, data=data), n_trials=50)

print(f"Best hyperparameters: {study.best_params}\n")
print(f"Best validation accuracy: {study.best_value}\n")

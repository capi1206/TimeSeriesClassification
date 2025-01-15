import numpy as np
import torch

def norm_mts_with_window(data, window_size=30):
    """
    Normalize a multivariate time series so that each component of the vector has
    a standard deviation of 1 within a sliding window lokking back in time.
    If window size is 1 does nothing.

    Returns np.array normalized time series data with the same shape as input.
    """
    t_steps, _ = data.shape
    normalized_data = np.zeros_like(data)

    for t in range(t_steps):
        start_i = max(0, t - window_size + 1)
        end_i = t + 1
        window = data[start_i:end_i]
        std = np.std(window, axis=0)
        mean = np.mean(window, axis=0)
        std[std == 0] = 1
        normalized_data[t] = (data[t])/ std

    return normalized_data


#function to create tensor with seg_length of backward steps
def create_seq_from_ts(data, seq_length):
    """
    Function that returns the vectors with corresponding seq_length,
    of observations in the past.

    Output should have shape (data.shape[0] - seq_length, seq_length, num_features)

    """
    sequences = []

    for i in range(len(data) - seq_length + 1):
        # Extract the sequence of features
        sequence = data[i:i+seq_length]
        sequences.append(sequence)

    sequences = np.array(sequences)

    return sequences

def prepare_data_random(data, sequence_length=30, val_perc = 0.1, test_perc = 0.05):
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
        sequences = create_seq_from_ts(data_2_norm, sequence_length)
        if np.isnan(sequences).any().any():
            print(f"Something wrong at sequences in {j}")
            continue

        # Prepare targets
        y = xdata[sequence_length - 1:, -1]

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
    return ((data_tr_x, data_tr_y), 
            (data_val_x, data_val_y),
            (data_tst_x, data_tst_y))
    
def shuffle_data(X_data, y_data):
    """ With an input tensor (X,y) returns the tensor keeping X-y relations
        with the indices shuffled
    """  
    X_data = torch.Tensor(X_data)
    y_data = torch.Tensor(y_data)   
    n_samples = X_data.size(0)
    indices = torch.randperm(n_samples)
    X_shuffled = X_data[indices]
    y_shuffled = y_data[indices]
    return X_shuffled, y_shuffled    

def print_progress(e, epochs, 
                   train_loss, train_accuracy,
                   val_loss, val_accuracy):
        print(f"\n Epoch {e + 1}/{epochs}:\n")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f" Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        
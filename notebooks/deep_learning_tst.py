import os
import numpy as np
#import logging
import sys
import torch
import torch.nn as nn
import math
import random
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy

from data_deep_learning.transformer import Transformer
from data_deep_learning.utils import norm_mts_with_window, create_seq_from_ts


###------THE MODEL------------------


# class MultiHeadAttention(nn.Module):

#     def __init__(self, d_model, n_head, details):
#         super(MultiHeadAttention, self).__init__()
#         self.n_head = n_head
#         self.attention = ScaleDotProductAttention( details=details)
#         self.w_q = nn.Linear(d_model, d_model)
#         self.w_k = nn.Linear(d_model, d_model)
#         self.w_v = nn.Linear(d_model, d_model)
#         self.w_concat = nn.Linear(d_model, d_model)
#         self.details = details

#     def forward(self, q, k, v ):
#         # 1. dot product with weight matrices

#         q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

#         if self.details: print('in Multi Head Attention Q,K,V: '+ str(q.size()))
#         # 2. split tensor by number of heads
#         q, k, v = self.split(q), self.split(k), self.split(v)

#         if self.details: print('in splitted Multi Head Attention Q,K,V: '+ str(q.size()))
#         # 3. do scale dot product to compute similarity
#         out, attention = self.attention(q, k, v )

#         if self.details: print('in Multi Head Attention, score value size: '+ str(out.size()))
#         # 4. concat and pass to linear layer
#         out = self.concat(out)
#         out = self.w_concat(out)

#         # 5. visualize attention map
#         # TODO : we should implement visualization

#         if self.details: print('in Multi Head Attention, score value size after concat : '+ str(out.size()))
#         return out

#     def split(self, tensor):
#         """
#         split tensor by number of head

#         :param tensor: [batch_size, length, d_model]
#         :return: [batch_size, head, length, d_tensor]
#         """
#         batch_size, length, d_model = tensor.size()
#         d_tensor = d_model // self.n_head
#         tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
#         # it is similar with group convolution (split by number of heads)

#         return tensor

#     def concat(self, tensor):
#         """
#         inverse function of self.split(tensor : torch.Tensor)

#         :param tensor: [batch_size, head, length, d_tensor]
#         :return: [batch_size, length, d_model]
#         """
#         batch_size, head, length, d_tensor = tensor.size()
#         d_model = head * d_tensor

#         tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
#         return tensor




# class ScaleDotProductAttention(nn.Module):
#     """
#     compute scale dot product attention

#     Query : given sentence that we focused on (decoder)
#     Key : every sentence to check relationship with Qeury(encoder)
#     Value : every sentence same with Key (encoder)
#     """

#     def __init__(self, details):
#         super(ScaleDotProductAttention, self).__init__()
#         self.softmax = nn.Softmax(dim=-1)
#         self.details = details
#     def forward(self, q, k, v ,e=1e-12):
#         # input is 4 dimension tensor
#         # [batch_size, head, length, d_tensor]
#         batch_size, head, length, d_tensor = k.size()

#         # 1. dot product Query with Key^T to compute similarity
#         k_t = k.transpose(2, 3)  # transpose

#         if self.details: print('in Scale Dot Product, k_t size: '+ str(k_t.size()))
#         score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product


#         if self.details: print('in Scale Dot Product, score size: '+ str(score.size()))
#         # 3. pass them softmax to make [0, 1] range
#         score = self.softmax(score)

#         if self.details: print('in Scale Dot Product, score size after softmax : '+ str(score.size()))

#         if self.details: print('in Scale Dot Product, v size: '+ str(v.size()))
#         # 4. multiply with Value
#         v = score @ v

#         if self.details: print('in Scale Dot Product, v size after matmul: '+ str(v.size()))
#         return v, score

# class PositionwiseFeedForward(nn.Module):

#     def __init__(self, d_model, hidden, drop_prob=0.1):
#         super(PositionwiseFeedForward, self).__init__()
#         self.linear1 = nn.Linear(d_model, hidden)
#         self.linear2 = nn.Linear(hidden, d_model)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=drop_prob)

#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.linear2(x)
#         return x

# class LayerNorm(nn.Module):
#     def __init__(self, d_model, eps=1e-12):
#         super(LayerNorm, self).__init__()
#         self.gamma = nn.Parameter(torch.ones(d_model))
#         self.beta = nn.Parameter(torch.zeros(d_model))
#         self.eps = eps

#     def forward(self, x):
#         mean = x.mean(-1, keepdim=True)
#         var = x.var(-1, unbiased=False, keepdim=True)
#         # '-1' means last dimension.

#         out = (x - mean) / torch.sqrt(var + self.eps)
#         out = self.gamma * out + self.beta
#         return out
    


# class Transformer(nn.Module):

#     def __init__(self,device, d_model=100, n_head=4, max_len=500, seq_len=30,
#                  ffn_hidden=128, n_layers=4, drop_prob=0.1, details =False):
#         super().__init__()
#         self.device = device
#         self.details = details
#         self.encoder_input_layer = nn.Linear(
#             in_features=145,
#             out_features=d_model
#             )

#         self.pos_emb = PostionalEncoding( max_seq_len=max_len,batch_first=False, d_model=d_model, dropout=0.1) #try different values of drupout?
#         self.encoder = Encoder(d_model=d_model,
#                                n_head=n_head,
#                                ffn_hidden=ffn_hidden,
#                                drop_prob=drop_prob,
#                                n_layers=n_layers,
#                                details=details,
#                                device=device)
#         self.classHead = ClassificationHead(seq_len=seq_len,d_model=d_model,details=details,n_classes=1)

#     def forward(self, src ):
#         if self.details: print('before input layer: '+ str(src.size()) )
#         src= self.encoder_input_layer(src)
#         if self.details: print('after input layer: '+ str(src.size()) )
#         src= self.pos_emb(src)
#         if self.details: print('after pos_emb: '+ str(src.size()) )
#         enc_src = self.encoder(src)
#         cls_res = self.classHead(enc_src)
#         if self.details: print('after cls_res: '+ str(cls_res.size()) )
#         return cls_res



# class PostionalEncoding(nn.Module):
#     def __init__(
#         self,
#         dropout: float=0.1,
#         max_seq_len: int=5000,
#         d_model: int=512,
#         batch_first: bool=False    ):
#         super().__init__()

#         self.d_model = d_model
#         self.dropout = nn.Dropout(p=dropout)
#         self.batch_first = batch_first
#         self.x_dim = 1 if batch_first else 0
#         position = torch.arange(max_seq_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_seq_len, 1, d_model)

#         pe[:, 0, 0::2] = torch.sin(position * div_term)

#         pe[:, 0, 1::2] = torch.cos(position * div_term)

#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(self.x_dim)]

#         x = self.dropout(x)
#         return x


# class EncoderLayer(nn.Module):

#     def __init__(self, d_model, ffn_hidden, n_head, drop_prob,details):
#         super(EncoderLayer, self).__init__()
#         self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head, details=details)
#         self.norm1 = LayerNorm(d_model=d_model)
#         self.dropout1 = nn.Dropout(p=drop_prob)
#         self.details = details
#         self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
#         self.norm2 = LayerNorm(d_model=d_model)
#         self.dropout2 = nn.Dropout(p=drop_prob)

#     def forward(self, x):
#         # 1. compute self attention
#         _x = x
#         x = self.attention(q=x, k=x, v=x )

#         if self.details: print('in encoder layer : '+ str(x.size()))
#         # 2. add and norm
#         x = self.dropout1(x)
#         x = self.norm1(x + _x)

#         if self.details: print('in encoder after norm layer : '+ str(x.size()))
#         # 3. positionwise feed forward network
#         _x = x
#         x = self.ffn(x)

#         if self.details: print('in encoder after ffn : '+ str(x.size()))
#         # 4. add and norm
#         x = self.dropout2(x)
#         x = self.norm2(x + _x)
#         return x

# class Encoder(nn.Module):

#     def __init__(self, d_model, ffn_hidden, n_head, n_layers,
#                  drop_prob, details, device):
#         super().__init__()


#         self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
#                                                   ffn_hidden=ffn_hidden,
#                                                   n_head=n_head
#                                                   ,details=details,
#                                                   drop_prob=drop_prob)
#                                      for _ in range(n_layers)])

#     def forward(self, x ):
#         for layer in self.layers:
#             x = layer(x)
#         return x


# class ClassificationHead(nn.Module):
#     def __init__(self,d_model, seq_len , details, n_classes: int = 1):
#       super().__init__()
#       self.norm = nn.LayerNorm(d_model)
#       self.details = details
#       #self.flatten = nn.Flatten()
#       self.seq = nn.Sequential( nn.Flatten() , nn.Linear(d_model * seq_len , 512) ,nn.ReLU(),nn.Linear(512, 256)
#                                ,nn.ReLU(),nn.Linear(256, 128),nn.ReLU(),nn.Linear(128, n_classes))

#     def forward(self,x):

#       if self.details:  print('in classification head : '+ str(x.size()))
#       x= self.norm(x)
#       #x= self.flatten(x)
#       x= self.seq(x)
#       if self.details: print('in classification head after seq: '+ str(x.size()))
#       return x    


# def norm_mts_with_window(data, window_size=30):
#     """
#     Normalize a multivariate time series so that each component of the vector has
#     a standard deviation of 1 within a sliding window lokking back in time.
#     If window size is 1 does nothing.

#     Returns np.array normalized time series data with the same shape as input.
#     """
#     t_steps, num_features = data.shape
#     normalized_data = np.zeros_like(data)

#     for t in range(t_steps):
#         start_i = max(0, t - window_size + 1)
#         end_i = t + 1
#         window = data[start_i:end_i]
#         std = np.std(window, axis=0)
#         mean = np.mean(window, axis=0)
#         std[std == 0] = 1
#         normalized_data[t] = (data[t])/ std

#     return normalized_data


#function to create tensor with seg_length of backward steps
# def create_seq_from_ts(data, seq_length):
#     """
#     Function that returns the vectors with corresponding seq_length,
#     of observations in the past.

#     Output should have shape (data.shape[0] - seq_length, seq_length, num_features)

#     """
#     sequences = []

#     for i in range(len(data) - seq_length + 1):
#         # Extract the sequence of features
#         sequence = data[i:i+seq_length]
#         sequences.append(sequence)

#     sequences = np.array(sequences)

#     return sequences


tst_perc = 0.05
val_perc = 0.1

seq_length = 32
d_model = 80
n_head = 2
max_len = 3000
ffn_hidden = 60
n_layers = 2
n_classes = 1
drop_prob = 0.4
details = False
epochs = 100
batch_size = 32
patience = 14
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(device=device,
                    d_model=d_model,
                    n_head=n_head,
                    max_len=max_len,
                    seq_len=seq_length,
                    ffn_hidden=ffn_hidden,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    details=details
                    ).to(device) 
model.load_state_dict(torch.load("btc_model_weights_params.pth"))
visited = ['BTC/USDT'] #'DUSK/USDT', 'XMR/USDT', 'MATIC/USDT', 'HOT/USDT', 'DOGE/USDT', 'BAND/USDT', 'LINK/USDT', 'REN/USDT', 'WAVES/USDT', 'CVC/USDT', 'ZRX/USDT', 'IOTX/USDT', 'ONT/USDT', 'TOMO/USDT', 'CELR/USDT', 'BTC/USDT', 'DENT/USDT', 'RVN/USDT']
'''logging.basicConfig(
    filename="progress.log",  # File where logs will be saved
    level=logging.INFO,       # Log level (e.g., DEBUG, INFO, WARNING, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S"  # Timestamp format
    import sys

# Redirect stdout to a file

)'''

# Define the directory containing the .npy files
directory_path = "data_deep_learning/series_data"

count_f = 0
with open("trial.log", "w") as f:
    sys.stdout = f

    
 
    for file_name in os.listdir(directory_path):
        
        if count_f > 4 :
            print("Process interrumpted")
            
            break
        
        if file_name.endswith(".npy") and ("ETH" in file_name 
                                           or 'DOGE' in file_name or 'LTC' in file_name
                                           or 'XRP' in file_name):  
            key = file_name.split('.')[0].replace("&","/")
            count_f +=1
            if not key in visited:
                print(f"Processing series {key}...")
            
                data = np.load(directory_path+"/"+file_name)
                X=data[-6000:,:-1]
                y=data[-6000:,-1]
                del data
                #X=X[-3000:, 72:]
                #y=y[-3000:]
                
                n_X = norm_mts_with_window(X, window_size=30)    
                n_X = create_seq_from_ts(n_X, seq_length)
                print(f"norm and turning into sequences complete.")
                n_y=y[seq_length-1:]    
                
                ind_tst = int(n_X.shape[0] * tst_perc)
                ind_val = int(n_X.shape[0] * val_perc)
                
                X_tr, X_val, X_tst = n_X[: -(ind_tst + ind_val),:,:], n_X[-(ind_tst + ind_val):-ind_tst,:,:], n_X[-ind_tst:,:,:] 
                y_tr, y_val, y_tst = n_y[: -(ind_tst + ind_val)], n_y[-(ind_tst + ind_val):-ind_tst], n_y[-ind_tst:] 

                X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
                y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
                X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
                y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
                X_tst_t = torch.tensor(X_tst, dtype=torch.float32).to(device)
                y_tst_t = torch.tensor(y_tst, dtype=torch.float32).to(device)

                #scale targets to lie between 1 and 0
                y_tr_t = (y_tr_t +1) / 2
                y_val_t = (y_val_t +1) / 2
                y_tst_t = (y_tst_t +1) / 2
                
                # SEED = 77
                # random.seed(SEED)
                # np.random.seed(SEED)
                # torch.manual_seed(SEED)
                # if torch.cuda.is_available():
                #     torch.cuda.manual_seed(SEED)
                #     torch.cuda.manual_seed_all(SEED)
                #     torch.backends.cudnn.deterministic = True
                #     torch.backends.cudnn.benchmark = False


                
                best_v_acc = 0
                n_noimprov = 0
                best_model_params = None

                y_tr_tensor = torch.tensor(y_tr_t, dtype=torch.long)
                y_val_tensor = torch.tensor(y_val_t, dtype=torch.long)


                # Prepare data loaders
                train_dataset = TensorDataset(X_tr_t, y_tr_tensor)
                val_dataset = TensorDataset(X_val_t, y_val_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)

                # Calculate class weights
                pos_weight = len(y_tr_t) / sum(y_tr_t)  # Assumes y_tr_t is binary {0, 1}
                #neg_weight = len(y_tr_t) / (len(y_tr_t) - sum(y_tr_t))

                # Use a tensor to represent positive and negative weights
                #class_weight = torch.tensor([neg_weight, pos_weight]).to(device)

                # Update criterion
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                # Define loss function and optimizer
                #criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=1e-3)

                # Training loop
                print(f"------------------for series {key} training loop \n\n")
                for epoch in range(epochs):
                    model.train()
                    train_loss = 0.0
                    correct = 0
                    total = 0
                    
                    for X_batch, y_batch in train_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)

                        # Ensure y_batch is of the correct type
                        y_batch = y_batch.float()
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)

                        # Backward pass and optimization
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item() * X_batch.size(0)

                        # Compute predictions and accuracy
                        probabilities = torch.sigmoid(outputs)
                        predicted = (probabilities > 0.5).float()
                        correct += (predicted == y_batch).sum().item()
                        total += y_batch.size(0)

                    train_loss /= len(train_loader.dataset)
                    train_accuracy = 100 * correct / total
                    
                    # Validation loop
                    model.eval()
                    val_loss = 0.0
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
                            y_batch = y_batch.float()

                            outputs = model(X_batch)
                            loss = criterion(outputs, y_batch)

                            val_loss += loss.item() * X_batch.size(0)
                            probabilities = torch.sigmoid(outputs)
                            predicted = (probabilities > 0.5).float()
                            correct += (predicted == y_batch).sum().item()
                            total += y_batch.size(0)

                    val_loss /= len(val_loader.dataset)
                    val_accuracy = 100 * correct / total
                    
                    if train_accuracy > 90:
                        if val_accuracy > best_v_acc:
                            best_v_acc = val_accuracy
                            best_model_params = copy.deepcopy(model.state_dict())
                            n_noimprov = 0
                        else:
                            n_noimprov +=1
                                
                    if n_noimprov > patience and best_model_params: #best_model_params is not None
                        print(f"Early stopping triggered, best acc on val data {best_v_acc}")   
                        break 

                    print(f"Epoch {epoch + 1}/{epochs}, "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
                    
                    if epoch == epochs -1:
                        print(f"Model's val acc : {best_v_acc}")
                
                if best_model_params:
                    model.load_state_dict(best_model_params)
                
                print("Training complete.")
                visited.append(key)
    if model:
        torch.save(model.state_dict(), "btc_model_weights_params2.pth")
    print(f" visited = {visited}")     
        
sys.stdout = sys.__stdout__
print("Progress log has been written to progress.log")   
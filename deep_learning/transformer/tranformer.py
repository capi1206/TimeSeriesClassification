import torch.nn as nn
import torch

from .units import ClassificationHead, Encoder

class Transformer(nn.Module):

    def __init__(self,device, d_model=100, n_head=4, input_dim=145, seq_len=30,
                 ffn_hidden=128, n_layers=4, drop_prob=0.1, l1_lambda=0.01):
        super().__init__()
        self.device = device
        self.l1_lambda = l1_lambda
        
        self.encoder_input_layer = nn.Linear(
            in_features=input_dim,
            out_features=d_model
            )
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        self.class_head = ClassificationHead(seq_len=seq_len,d_model=d_model,n_classes=1)
    
        
    def forward(self, src ):
        
        src= self.encoder_input_layer(src)
        enc_src = self.encoder(src)
        cls_res = self.class_head(enc_src)
        return cls_res
    
    def l1_loss(self):

        l1_loss = torch.sum(torch.abs(self.encoder_input_layer.weight))
        return self.l1_lambda * l1_loss
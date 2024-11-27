import torch
from torch import cuda
import torch.nn as nn
from torch.nn import functional as F
from config import *
from utilities import convert_tokens_to_perm, token_type
import numpy as np
import re
from transformer import Transformer

class Probe(nn.Module):
    def __init__(self, in_shape):
        super().__init__()

        self.classifier = nn.Linear(in_shape[0]*in_shape[1], 2, bias=False)
    
    def forward(self, x):
        return self.classifier(x.flatten(start_dim=1))

class ProbedTransformer(Transformer):
    def __init__(self, stop_block=0):
        super().__init__()

        self.requires_grad_(False)

        self.stop_block = stop_block
        self.probe = Probe((CONTEXT_LENGTH, n_embed))

        self.blocks = self.blocks[:stop_block]
        self.blocks.append(self.probe)
    
    def forward(self, idx):
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) #(B, T, C)
        pos_emb = self.position_embedding(torch.arange(idx.shape[1], device=idx.device)) #(T, C)

        x = tok_emb + pos_emb #(B, T, C)
        x = self.embed_hook(x)

        x = self.sa_heads(x) # apply one head of self attention (B, T, C)
        logits = self.blocks(x) # (B, 2, 1)

        return logits
import torch
from torch import cuda
import torch.nn as nn
from torch.nn import functional as F
from config import *
from utilities import convert_tokens_to_perm
import numpy as np

# model based off of this video: https://www.youtube.com/watch?v=kCc8FmEb1nY
# and by "based off" i mean i modifed like two things
# basically i removed the masking from the self attention step
# and changed the output to be logisitic and one node instead of softmax

assert n_embed % n_head == 0 

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(
            self.proj(
                torch.cat([h(x) for h in self.heads], dim=-1)
            )
        )

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.ReLU(),
            nn.Linear(n_embed * 4, n_embed), # projection layer
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Commmunication followed by computation"""

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head

        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # residuals
        # don't use += as this breaks things
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)

        self.sa_heads = MultiHeadAttention(n_head, n_embed//n_head)
        self.blocks = [Block(n_embed, n_head) for _ in range(n_blocks)]
        self.blocks.append(nn.LayerNorm(n_embed))

        self.blocks = nn.Sequential(
            *self.blocks
        )

        # the output layer
        # projects the final vector down to the output dimension
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
       
        # we shouldn't use this during training, only generation
        # this is because cross entropy loss already applies a softmax
        # and we don't want to apply that twice
        self.softmax = nn.Softmax(dim=1)

    def forward(self, idx):
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) #(B, T, C)
        pos_emb = self.position_embedding(torch.arange(idx.shape[1], device=idx.device)) #(T, C)

        x = tok_emb + pos_emb #(B, T, C)
        x = self.sa_heads(x) # apply one head of self attention
        x = self.blocks(x) # apply a bunch of blocks (sa + feedforward)

        logits = self.lm_head(x) #(B, T, vocab_size)

        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)

        return logits
    
    def generate(self, sequence):
        self.eval()

        # use gpu for processing
        if cuda.is_available():
          dev = "cuda:0"
        else:
          dev = "cpu"
        
        # create an initial input to the network
        input_tensor = torch.ones(block_size, dtype=int).to(dev)
        input_tensor *= NULL_TOKEN
        input_tensor[:len(sequence)] = torch.tensor(sequence, dtype=int).to(dev)
        input_tensor[len(sequence)] = START_PREDICTION_TOKEN

        # tells us where the input ends and the autoregression starts
        AR_index = len(sequence) + 1
        
        # actually generate the permutation
        permutation = []

        # do the autoregression
        for x in range(MAX_GROUP_SIZE):
            # get the logits
            logits = self(input_tensor.unsqueeze(0))
            
            # get the most likely token
            token = torch.argmax(logits, dim=1).item()
            
            # append it to the permutation
            permutation.append(token)
            
            # add it to the input tensor
            input_tensor[AR_index + x] = token
        
        return np.array(convert_tokens_to_perm(permutation))
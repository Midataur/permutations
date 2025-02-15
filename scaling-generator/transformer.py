import torch
from torch import cuda
import torch.nn as nn
from torch.nn import functional as F
from config import *
from utilities import convert_tokens_to_perm, token_type
from accelerate import Accelerator
import numpy as np

# model based off of this video: https://www.youtube.com/watch?v=kCc8FmEb1nY
# and by "based off" i mean i modifed like two things (edit: a bit more than that by now)
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

        # create a mask that only effects the permutation tokens
        noninputlength = CONTEXT_LENGTH-INPUT_LENGTH
        A = torch.ones(INPUT_LENGTH, INPUT_LENGTH)
        B = torch.zeros(INPUT_LENGTH, noninputlength)
        C = torch.ones(noninputlength, INPUT_LENGTH)
        D = torch.tril(torch.ones(noninputlength, noninputlength))

        E = torch.cat((torch.cat((A, B), dim=1), torch.cat((C, D), dim=1)), dim=0)

        self.register_buffer('tril', E)

        # this is just a place to attach a hook
        self.attention_hook = nn.Identity()

        self.sanity_hook = nn.Identity()

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)

        if MASKED_MODEL:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)

        wei = self.sanity_hook(wei)

        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        wei = self.attention_hook(wei)

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

class Transformer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)

        # this is just a place to attach a hook
        self.embed_hook = nn.Identity()

        if LEGACY_ARCHITECTURE:
            self.sa_heads = MultiHeadAttention(n_head, n_embed//n_head)
        
        self.blocks = [Block(n_embed, n_head) for _ in range(n_blocks)]
        self.blocks.append(nn.LayerNorm(n_embed))

        self.blocks = nn.Sequential(*self.blocks)

        # the output layer
        # projects the final vector down to the output dimension
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
       
        # we shouldn't use this during training, only generation
        # this is because cross entropy loss already applies a softmax
        # and we don't want to apply that twice
        self.softmax = nn.Softmax(dim=1)

    def forward(self, idx):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) #(B, T, C)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device)) #(T, C)

        x = tok_emb + pos_emb #(B, T, C)
        x = self.embed_hook(x)

        if LEGACY_ARCHITECTURE:
            x = self.sa_heads(x) # apply one head of self attention (B, T, C)
        
        x = self.blocks(x) # apply a bunch of blocks (sa + feedforward) (B, T, C)

        logits = self.lm_head(x) #(B, T, vocab_size)

        # focus only on the last time step
        
        if REVERSE_PROBLEM:
            logits = logits[:, -INPUT_LENGTH:, :]
        elif MASKED_MODEL:
            logits = logits[:, -MAX_GROUP_SIZE:, :]
        else:
            # deprecated
            logits = logits[:, -1, :] # becomes (B, C)

        return logits
    
    def generate(self, sequence, accelerator, force_valid=False, debug=False, stop_at=float("inf")):
        """
            Generates a permutation for a sequence.
            If force_valid is set to True then the sequence is
            guaranteed to be a valid permutation, if not a correct one.
        """

        dev = accelerator.device

        # handle old models
        if not MASKED_MODEL:
            return self.old_generate(sequence, force_valid, debug, stop_at)

        self.eval()
        
        # create an initial input to the network
        input_tensor = torch.ones(INPUT_LENGTH + 1, dtype=int).to(dev)
        input_tensor[:len(sequence)] = torch.tensor(sequence, dtype=int).to(dev)
        input_tensor[-1] = START_PREDICTION_TOKEN
        
        # actually generate the permutation
        permutation = []

        # print("Model device:", self.token_embedding_table.weight.get_device())
        # print("Input tensor device:", input_tensor.get_device())

        # do the autoregression
        for x in range(MAX_GROUP_SIZE):
            # get the logits
            logits = self(input_tensor.unsqueeze(0))[:, -1, :]

            if debug:
                print(f"Step {x} logits: {logits}")
            
            # get the most likely token
            chosen = None

            if not force_valid:
                chosen = logits.argmax().item()
            else:
                for token in logits.argsort(descending=True)[0]:
                    if token not in permutation and token_type(token) == "permutation":
                        chosen = token
                        break
            
            if chosen == None:
                raise Exception("No possible token found - this shouldn't be possible")
            
            # append it to the permutation
            permutation.append(chosen)
            
            # add it to the input tensor
            input_tensor = torch.cat((input_tensor, torch.tensor([chosen]).to(dev)))

            # allows for early stopping
            if x >= stop_at:
                break
        
        return np.array(convert_tokens_to_perm(permutation))

    def old_generate(self, sequence, force_valid=False, debug=False, stop_at=float("inf")):

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

            if debug:
                print(f"Step {x} logits: {logits}")
            
            # get the most likely token
            chosen = None

            if not force_valid:
                chosen = logits.argmax().item()
            else:
                for token in logits.argsort(descending=True)[0]:
                    if token not in permutation and token_type(token) == "permutation":
                        chosen = token
                        break
            
            if chosen == None:
                raise Exception("No possible token found - this shouldn't be possible")
            
            # append it to the permutation
            permutation.append(chosen)
            
            # add it to the input tensor
            input_tensor[AR_index + x] = chosen

            # allows for early stopping
            if x >= stop_at:
                break
        
        return np.array(convert_tokens_to_perm(permutation))
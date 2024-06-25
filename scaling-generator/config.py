import torch
from math import log2

torch.manual_seed(42)

# GLOBAL
MAX_GROUP_SIZE = 16

if int(log2(MAX_GROUP_SIZE)) != log2(MAX_GROUP_SIZE):
    raise Exception("MAX_GROUP_SIZE must be a power of 2")

# the maximum number of transpositions in the input sequence
MAX_TRANS_NUMBER = 10
# maximum length of input sequence (in tokens)
# don't touch this
INPUT_LENGTH = (int(log2(MAX_GROUP_SIZE)) + 1) * 2 * MAX_TRANS_NUMBER
CONTEXT_LENGTH = INPUT_LENGTH + 1 + MAX_GROUP_SIZE + 1
PATH = "."
DATA = "/data/window_test/"
MODELNAME = "window-1.0"
# can be "full" or an integer
# i recommend 64
BATCHSIZE = 64

# general or elementary

# base to use for inputting transpositions
TRANS_BASE = 2

# SPECIAL TOKENS
# do not change manually
num_normal = TRANS_BASE + MAX_GROUP_SIZE
num_special = 3
NULL_TOKEN = num_normal
START_PREDICTION_TOKEN = num_normal + 1
END_PREDICTION_TOKEN = num_normal + 2

# TRANSFORMER HYPERPARAMETERS
vocab_size = num_normal + num_special
n_embed = 102
block_size = CONTEXT_LENGTH
n_head = 6
n_blocks = 4
dropout = 0

# TRAINING HYPERPARAMETERS
# good starting value: 3*10^-5
learning_rate = 3*(10**-4)
num_epochs = 10**8

# good starting value: 0.01
weight_decay = 0.005

lr_factor = 0.1  # Factor by which the learning rate will be reduced
lr_patience = 10  # Number of epochs with no improvement after which learning rate will be reduced
threshold = 0.01  # Threshold for measuring the new optimum
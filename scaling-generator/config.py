import torch

torch.manual_seed(42)

# GLOBAL
MAX_GROUP_SIZE = 4
# the maximum length of the input sequence
# this is not the number of transpositions
# it is the space it takes up in the context length
MAX_INPUT_LENGTH = 12
CONTEXT_LENGTH = MAX_INPUT_LENGTH + 1 + MAX_GROUP_SIZE + 1
PATH = "."
DATA = "/data/small/"
MODELNAME = "small-6.0"
# can be "full" or an integer
# i recommend 64
BATCHSIZE = 64

# general or elementary

# base to use for inputting transpositions
TRANS_BASE = 10 # using base 10 for convenience, probably not optimal

# SPECIAL TOKENS
# do not change manually
num_normal = TRANS_BASE + MAX_GROUP_SIZE
num_special = 3
NULL_TOKEN = num_normal
START_PREDICTION_TOKEN = num_normal + 1
END_PREDICTION_TOKEN = num_normal + 2

# TRANSFORMER HYPERPARAMETERS
vocab_size = num_normal + num_special
n_embed = vocab_size + 1 # 18
block_size = CONTEXT_LENGTH
n_head = 6
n_blocks = 8
dropout = 0

# TRAINING HYPERPARAMETERS
# good starting value: 3*10^-5
learning_rate = 3*(10**-5)
num_epochs = 10**8

# good starting value: 0.01
weight_decay = 0.01

lr_factor = 0.1  # Factor by which the learning rate will be reduced
lr_patience = 10  # Number of epochs with no improvement after which learning rate will be reduced
threshold = 0.01  # Threshold for measuring the new optimum
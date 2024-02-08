# GLOBAL
MAX_LENGTH = 6
GROUP_SIZE = 4
IDENTITY_PROPORTION = 0.5 # controls what proportion of the training data is artifically added identities
PATH = "."
DATA = "/data/forcegrok/"
MODELNAME = "forcegrok14.0"
# can be "full" or an integer
# i recommend 64
BATCHSIZE = 64

# general or elementary
TRANSPOSITION_TYPE = "elementary"

# TRANSFORMER HYPERPARAMETERS
n_embed = 384
vocab_size = GROUP_SIZE**2 if TRANSPOSITION_TYPE == "general" else GROUP_SIZE
block_size = MAX_LENGTH
n_head = 6
n_blocks = 1
dropout = 0

# TRAINING HYPERPARAMETERS
# good starting value: 3*10^-5
learning_rate = 3*(10**-5)
num_epochs = 10**8

# good starting value: 0.01
weight_decay = 1

lr_factor = 0.1  # Factor by which the learning rate will be reduced
lr_patience = 10  # Number of epochs with no improvement after which learning rate will be reduced
threshold = 0.01  # Threshold for measuring the new optimum
# GLOBAL
MAX_LENGTH = 6
GROUP_SIZE = 4
PATH = "."
DATA = "/data/smallgeneral/"
MODELNAME = "smallgeneral2"
# can be "full" or an integer
# i recommend 64
BATCHSIZE = 64

# general or elementary
TRANSPOSITION_TYPE = "general"

# TRANSFORMER HYPERPARAMETERS
n_embed = 384
normal_tokens = GROUP_SIZE**2 if TRANSPOSITION_TYPE == "general" else GROUP_SIZE
vocab_size = normal_tokens + 2
block_size = MAX_LENGTH + 1 + GROUP_SIZE
n_head = 6
n_blocks = 4
dropout = 0

# TRAINING HYPERPARAMETERS
# good starting value: 3*10^-5
learning_rate = 3*(10**-6)
num_epochs = 10**8

# good starting value: 0.01
weight_decay = 0.01

lr_factor = 0.1  # Factor by which the learning rate will be reduced
lr_patience = 10  # Number of epochs with no improvement after which learning rate will be reduced
threshold = 0.01  # Threshold for measuring the new optimum

# CONSTANTS
# do not change manually
START_PREDICTION_TOKEN = normal_tokens
TO_PREDICT_TOKEN = normal_tokens + 1
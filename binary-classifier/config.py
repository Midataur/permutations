# GLOBAL
MAX_LENGTH = 10
GROUP_SIZE = 5
IDENTITY_PROPORTION = 0.5 # controls what proportion of the training data is artifically added identities
PATH = "."
DATA = "/data/medtrans/"
MODELNAME = "medtrans-5.0"
# can be full or an integer
# i recommend 64
BATCHSIZE = 64

# TRANSFORMER HYPERPARAMETERS
n_embed = 384
vocab_size = GROUP_SIZE**2
block_size = MAX_LENGTH
n_head = 6
n_blocks = 2
dropout = 0.1

# TRAINING HYPERPARAMETERS
# good starting value: 3*10^-5
learning_rate = 3*(10**-5)
num_epochs = 10**8

# good starting value: 0.01
weight_decay = 0.01

lr_factor = 0.1  # Factor by which the learning rate will be reduced
lr_patience = 10  # Number of epochs with no improvement after which learning rate will be reduced
threshold = 0.01  # Threshold for measuring the new optimum
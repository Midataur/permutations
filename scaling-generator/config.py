import torch
from math import log2, floor

torch.manual_seed(42)

MAX_GROUP_SIZE = 16
ACTUAL_GROUP_SIZE = 10
WINDOW = True

# used to enable legacy features that have been deprecated
# this is for backwards compatability reasons
LEGACY_OVERRIDE = True
# used for backwards compatability with an older version of datagen
# adds one to the number of digits used for a binary number
DIGIT_OVERRIDE = True

# the maximum number of transpositions in the input sequence
MAX_TRANS_NUMBER = 10

# can be elementary (one token per transposition, only adjacent transpositions allowed)
# can be general (one token per transposition, general transpositions allowed) 
# or digital (each tranposition is written in place value notation)
INPUT_TYPE = "digital"
# maximum length of input sequence (in tokens)
# don't touch this

# base to use for inputting transpositions (if using digital)
#should be None or an integer
TRANS_BASE = 2

if INPUT_TYPE == "digital":
    DIGITS_USED = floor(log2(MAX_GROUP_SIZE)/log2(TRANS_BASE))

    if DIGIT_OVERRIDE:
        DIGITS_USED += 1

    INPUT_LENGTH = DIGITS_USED * 2 * MAX_TRANS_NUMBER

else:
    INPUT_LENGTH = MAX_TRANS_NUMBER

CONTEXT_LENGTH = INPUT_LENGTH + 1 + MAX_GROUP_SIZE

if LEGACY_OVERRIDE:
    CONTEXT_LENGTH += 1 # allows for deprecated end_sequence_token

PATH = "."
DATA = "/data/window_test/"
MODELNAME = "window-7.0"
# can be "full" or an integer
# i recommend 64
BATCHSIZE = 64

# TOKENS
# do not change unless you're max
match INPUT_TYPE:
    case "elementary":
        num_trans = MAX_GROUP_SIZE
    case "general":
        num_trans = MAX_GROUP_SIZE**2
    case "digital":
        num_trans = TRANS_BASE

num_normal = num_trans + MAX_GROUP_SIZE

num_special = 2
if LEGACY_OVERRIDE:
    num_special = 3

NULL_TOKEN = num_normal
START_PREDICTION_TOKEN = num_normal + 1

# this is deprecated and should no longer be used
END_PREDICTION_TOKEN = num_normal + 2

# TRANSFORMER HYPERPARAMETERS
# you can change these if you want
vocab_size = num_normal + num_special
n_embed = 402
block_size = CONTEXT_LENGTH
n_head = 6
n_blocks = 4
dropout = 0

# TRAINING HYPERPARAMETERS
# good starting value: 3*10^-5
learning_rate = 3*(10**-4)
num_epochs = 10**8

# good starting value: 0.01
weight_decay = 0.01

lr_factor = 0.1  # Factor by which the learning rate will be reduced
lr_patience = 10  # Number of epochs with no improvement after which learning rate will be reduced
threshold = 0.01  # Threshold for measuring the new optimum
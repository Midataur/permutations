from math import log2, floor
from accelerate.utils import set_seed
import torch

set_seed(42)

MAX_GROUP_SIZE = 16
ACTUAL_GROUP_SIZE = 10

WINDOW = True
WINDOW_COUNT = None
PARTITIONED_WINDOWS = True
RELABEL = False

PATH = "."

DATA = "/data/reverse/"
MODELNAME = "reverse-4"

# used to enable legacy features that have been deprecated
# this is for backwards compatability reasons
LEGACY_OVERRIDE = False

# used for backwards compatability with an older version of datagen
# adds one to the number of digits used for a binary number
DIGIT_OVERRIDE = False

# used for backwards compatability with unmasked models
# should always be true for new models
MASKED_MODEL = True

# are we doing the reverse problem?
# ie. predict the reduced word from the permutation
REVERSE_PROBLEM = True

# used for backwards compatability with older models
# we used to use a slightly weird non-standard architecture
LEGACY_ARCHITECTURE = False

# the maximum number of transpositions in the input sequence
MAX_TRANS_NUMBER = 120

# can be elementary (one token per transposition, only adjacent transpositions allowed)
# can be general (one token per transposition, general transpositions allowed)
# can be hybrid (two tokens per transposition, general transpositions allowed)
# or binary (each tranposition is written in binary)
INPUT_TYPE = "elementary"

# maximum length of input sequence (in tokens)
# don't touch this
if INPUT_TYPE == "binary":
    DIGITS_USED = floor(log2(MAX_GROUP_SIZE))

    if DIGIT_OVERRIDE:
        DIGITS_USED += 1

    INPUT_LENGTH = DIGITS_USED * 2 * MAX_TRANS_NUMBER
elif INPUT_TYPE == "hybrid":
    INPUT_LENGTH = MAX_TRANS_NUMBER * 2
else:
    INPUT_LENGTH = MAX_TRANS_NUMBER

CONTEXT_LENGTH = INPUT_LENGTH + MAX_GROUP_SIZE

if LEGACY_OVERRIDE:
    CONTEXT_LENGTH += 1 # allows for deprecated end_sequence_token

if not MASKED_MODEL:
    CONTEXT_LENGTH += 1 # allows for deprecated final permutation token

# TOKENS
# do not change unless you're max
if INPUT_TYPE in ["elementary", "hybrid"]:
    num_trans = MAX_GROUP_SIZE
elif INPUT_TYPE == "general":
    num_trans = MAX_GROUP_SIZE**2
elif INPUT_TYPE == "binary":
    num_trans = 2

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
n_blocks = 10
dropout = 0

# TRAINING HYPERPARAMETERS
# good starting value: 3*10^-5
learning_rate = 3*(10**-4)
num_epochs = 10**8
# can be "full" or an integer
# good starting value: 64
BATCHSIZE = 1024

# good starting value: 0.01
weight_decay = 0.01

lr_factor = 0.1  # Factor by which the learning rate will be reduced
lr_patience = 10  # Number of epochs with no improvement after which learning rate will be reduced
threshold = 0.01  # Threshold for measuring the new optimum

# for dataloading
N_WORKERS = 0
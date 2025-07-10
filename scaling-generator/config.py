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

DATA = "/data/elem-partitioned-long-boosted/"
MODELNAME = "elem-partitioned-long-boosted-1"

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
REVERSE_PROBLEM = False

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
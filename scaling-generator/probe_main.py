print("Loading libraries...")

from training import train
from config import *

# check if the last element is unchanged
def question(permutation):
    return permutation[-1] == MAX_GROUP_SIZE-1

if __name__ == "__main__":
    train()
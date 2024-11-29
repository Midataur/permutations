print("Loading libraries...")

from training import train
from probed_transfomer import ProbedTransformer
from dataloading import ProbeDataset
from config import *

# check if the last element is unchanged
def question(permutation):
    return int(permutation[-1] == MAX_GROUP_SIZE-1)

if __name__ == "__main__":
    train(
        model_class=ProbedTransformer,
        dataset_class=ProbeDataset,
        question=question,
        stop_block=2,
        suffix="-probed"
    )
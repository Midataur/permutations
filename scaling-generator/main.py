print("Loading libraries...")

from training import train
from config import MASKED_MODEL, REVERSE_PROBLEM
from dataloading import MaskedDataset, SimpleDataset, ReversedDataset

if __name__ == "__main__":
    if not MASKED_MODEL:
        dataset_class = SimpleDataset
    elif REVERSE_PROBLEM:
        dataset_class = ReversedDataset
    else:
        dataset_class = MaskedDataset

    train(dataset_class=dataset_class)
print("Loading libraries...")

from training import train
from config import MASKED_MODEL
from dataloading import MaskedDataset, SimpleDataset

if __name__ == "__main__":
    datasetClass = MaskedDataset if MASKED_MODEL else SimpleDataset
    train(dataset_class=datasetClass)
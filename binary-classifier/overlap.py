from utilities import *
from config import *
from dataloading import *
from tqdm import tqdm

# incomplete

# put training data into a set
train_set = set()

for targets, _ in tqdm(train_loader):
    for target in targets:
        train_set.add(target)
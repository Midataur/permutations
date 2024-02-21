from utilities import *
from config import *
from torch.utils.data import DataLoader, Dataset
from torch import tensor, float32, cuda, device
from scipy import sparse
from tqdm import tqdm
import os

class SimpleDataset(Dataset):
    def __init__(self, sequences):
        # use gpu for processing
        if cuda.is_available():
          dev = "cuda:0"
        else:
          dev = "cpu"

        dev = device(dev)

        if type(sequences) == sparse._csr.csr_matrix:
          sequences = sequences.todense()

        data = []
        targets = []

        for sequence in tqdm(sequences, desc="Loading data"):
          permutation = get_permutation(sequence)
          # word + start pred token + permutation
          new_seq = list(sequence) + [START_PREDICTION_TOKEN] + [TO_PREDICT_TOKEN for x in range(GROUP_SIZE)]
          for pos, char in enumerate(permutation):
            data.append(list(new_seq))

            new_seq[MAX_LENGTH + 1 + pos] = char

            target = [0 for x in range(vocab_size)]
            target[char] = 1

            targets.append(target)

        self.data = tensor(data, dtype=int).to(dev)
        self.targets = tensor(targets, dtype=float32).to(dev)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = (
            self.data[index],
            self.targets[index]
        )
        return sample

# load the data
print("Loading data...")

train_data = np.array([[0 for x in range(MAX_LENGTH)]])

curfile = 1
while True:
  filename = PATH + DATA + f"train_data{curfile}.csv"

  if not os.path.isfile(filename):
     break

  data = np.loadtxt(filename, delimiter=",").astype(int)

  train_data = np.concatenate((train_data, data))

  curfile += 1

train_data = train_data[1:]
DATASET_SIZE = len(train_data)

val_data = np.loadtxt(PATH + DATA + "val_data.csv", delimiter=",").astype(int)
test_data = np.loadtxt(PATH + DATA + "test_data.csv", delimiter=",").astype(int)

# get number of identities in true_test_data
# print("True identity count:", len(true_test_data[np.apply_along_axis(is_identity, 1, true_test_data)]))

# label the datasets
X_train = train_data
X_val = val_data
X_test = test_data

final_dimension = MAX_LENGTH

train_batch_size = BATCHSIZE if BATCHSIZE != "full" else len(X_train)*GROUP_SIZE
to_shuffle = BATCHSIZE != "full"

# create the dataloaders
train_dataset = SimpleDataset(X_train)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=to_shuffle, num_workers=0)

val_dataset = SimpleDataset(X_val)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

test_dataset = SimpleDataset(X_test)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
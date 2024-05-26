from utilities import *
from config import *
from torch.utils.data import DataLoader, Dataset
from torch import tensor, float32, cuda, device
from scipy import sparse
from tqdm import tqdm
import os

class SimpleDataset(Dataset):
    def __init__(self, sequences, permutations):
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

        # generate the input output pairs
        # we're basically simulating what the autoregression process would look like
        # this massively improves the effective amount of training data
        for sequence, permutation in tqdm(zip(sequences, permutations), desc="Loading data"):
          # shift the permutation to use the correct tokens
          # we don't want overlap between the input tokens and the output tokens
          shifted_perm = convert_perm_to_tokens(permutation)

          padding_len = CONTEXT_LENGTH - len(sequence) - 1

          # baseline input
          new_seq = list(sequence) + [START_PREDICTION_TOKEN] + [NULL_TOKEN]*(padding_len)

          # do the fake autoregression
          for pos, char in enumerate(shifted_perm + [END_PREDICTION_TOKEN]):
            data.append(list(new_seq))

            new_seq[len(sequence) + 1 + pos] = char

            # cross entropy loss prefers accepting a token
            # it's faster
            targets.append(char)

        self.data = tensor(data, dtype=int).to(dev)
        self.targets = tensor(targets, dtype=int).to(dev)

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

train_inputs = np.array([[0 for x in range(INPUT_LENGTH)]])
train_perms = np.array([[0 for x in range(MAX_GROUP_SIZE)]])

curfile = 1
while True:
  filename = PATH + DATA + f"train_data{curfile}.csv"
  perm_filename = PATH + DATA + f"train_data{curfile}_perms.csv"

  if not os.path.isfile(filename):
     break

  data = np.loadtxt(filename, delimiter=",").astype(int)
  perms = np.loadtxt(perm_filename, delimiter=",").astype(int)

  print("wahoo",len(data[0]), INPUT_LENGTH)

  train_inputs = np.concatenate((train_inputs, data))
  train_perms = np.concatenate((train_perms, perms))

  curfile += 1

train_inputs = train_inputs[1:]
train_perms = train_perms[1:]
DATASET_SIZE = len(train_inputs)

val_seqs = np.loadtxt(PATH + DATA + "val_data.csv", delimiter=",").astype(int)
val_perms = np.loadtxt(PATH + DATA + "val_data_perms.csv", delimiter=",").astype(int)

test_seqs = np.loadtxt(PATH + DATA + "test_data.csv", delimiter=",").astype(int)
test_perms = np.loadtxt(PATH + DATA + "test_data_perms.csv", delimiter=",").astype(int)

# create the dataloaders
train_dataset = SimpleDataset(train_inputs, train_perms)
train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, num_workers=0)

val_dataset = SimpleDataset(val_seqs, val_perms)
val_dataloader = DataLoader(val_dataset, batch_size=BATCHSIZE, num_workers=0)

test_dataset = SimpleDataset(test_seqs, test_perms)
test_dataloader = DataLoader(test_dataset, batch_size=BATCHSIZE, num_workers=0)
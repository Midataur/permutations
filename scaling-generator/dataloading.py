from utilities import *
from config import *
from torch.utils.data import DataLoader, Dataset
from torch import tensor, float32, cuda, device
from scipy import sparse
from tqdm import tqdm
from accelerate import Accelerator
import os

class SimpleDataset(Dataset):
    def __init__(self, sequences, permutations, accelerator):
        # use gpu for processing
        dev = accelerator.device

        if type(sequences) == sparse._csr.csr_matrix:
          sequences = sequences.todense()

        data = []
        targets = []

        # generate the input output pairs
        # we're basically simulating what the autoregression process would look like
        for sequence, permutation in tqdm(zip(sequences, permutations), desc="Loading data"):
          # shift the permutation to use the correct tokens
          # we don't want overlap between the input tokens and the output tokens
          shifted_perm = convert_perm_to_tokens(permutation)

          padding_len = CONTEXT_LENGTH - len(sequence) - 1

          # baseline input
          new_seq = list(sequence) + [START_PREDICTION_TOKEN] + [NULL_TOKEN]*(padding_len)

          # do the fake autoregression
          adding = shifted_perm

          if LEGACY_OVERRIDE:
            adding += [END_PREDICTION_TOKEN]

          for pos, char in enumerate(adding):
            data.append(list(new_seq))

            new_seq[len(sequence) + 1 + pos] = char

            # cross entropy loss prefers accepting a token
            # it's faster
            targets.append(char)

        self.data = tensor(data, dtype=int)
        self.targets = tensor(targets, dtype=int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = (
            self.data[index],
            self.targets[index]
        )
        return sample

def load_data(accelerator):
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

    train_inputs = np.concatenate((train_inputs, data))
    train_perms = np.concatenate((train_perms, perms))

    curfile += 1

  train_inputs = train_inputs[1:]
  train_perms = train_perms[1:]
  dataset_size = len(train_inputs)

  val_seqs = np.loadtxt(PATH + DATA + "val_data.csv", delimiter=",").astype(int)
  val_perms = np.loadtxt(PATH + DATA + "val_data_perms.csv", delimiter=",").astype(int)

  test_seqs = np.loadtxt(PATH + DATA + "test_data.csv", delimiter=",").astype(int)
  test_perms = np.loadtxt(PATH + DATA + "test_data_perms.csv", delimiter=",").astype(int)

  # create the dataloaders
  train_dataset = SimpleDataset(train_inputs, train_perms, accelerator)
  train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, num_workers=N_WORKERS)

  val_dataset = SimpleDataset(val_seqs, val_perms, accelerator)
  val_dataloader = DataLoader(val_dataset, batch_size=BATCHSIZE, num_workers=N_WORKERS)

  test_dataset = SimpleDataset(test_seqs, test_perms, accelerator)
  test_dataloader = DataLoader(test_dataset, batch_size=BATCHSIZE, num_workers=N_WORKERS)

  return (
     train_inputs, train_perms, train_dataloader, 
     val_seqs, val_perms, val_dataloader,
     test_seqs, test_perms, test_dataloader,
     dataset_size
  )
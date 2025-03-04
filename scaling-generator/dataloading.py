from utilities import *
from config import *
from torch.utils.data import DataLoader, Dataset
from torch import tensor, float32, cuda, device
from scipy import sparse
from tqdm import tqdm
from accelerate import Accelerator
import os

class SimpleDataset(Dataset):
    def __init__(self, sequences, permutations, *args, **kwargs):
        # use gpu for processing
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

class MaskedDataset(Dataset):
    def __init__(self, sequences, permutations, mainthread, *args, **kwargs):
        # use gpu for processing
        if type(sequences) == sparse._csr.csr_matrix:
          sequences = sequences.todense()

        data = []
        targets = []

        # generate the input output pairs
        # we're basically simulating what the autoregression process would look like
        for sequence, permutation in tqdm(
           zip(sequences, permutations), 
           desc="Loading data", 
           disable=not mainthread
        ):
          # shift the permutation to use the correct tokens
          # we don't want overlap between the input tokens and the output tokens
          shifted_perm = convert_perm_to_tokens(permutation)

          model_input = list(sequence) + [START_PREDICTION_TOKEN] + list(shifted_perm)[:-1]

          # create target
          target = shifted_perm

          data.append(model_input)
          targets.append(target) 

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

class ReversedDataset(Dataset):
    def __init__(self, sequences, permutations, mainthread, *args, **kwargs):
        # use gpu for processing
        if type(sequences) == sparse._csr.csr_matrix:
          sequences = sequences.todense()

        data = []
        targets = []

        # generate the input output pairs
        # we're basically simulating what the autoregression process would look like
        for sequence, permutation in tqdm(
           zip(sequences, permutations), 
           desc="Loading data", 
           disable=not mainthread
        ):
          # shift the permutation to use the correct tokens
          # we don't want overlap between the input tokens and the output tokens
          # we keep the tokens the same as in the normal problem to make things easier
          shifted_perm = convert_perm_to_tokens(permutation)

          model_input = list(shifted_perm) + [START_PREDICTION_TOKEN] + list(sequence)[:-1]

          # create target
          target = sequence

          data.append(model_input)
          targets.append(target) 

        self.data = tensor(np.array(data), dtype=int)
        self.targets = tensor(np.array(targets), dtype=int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = (
            self.data[index],
            self.targets[index]
        )
        return sample

class ProbeDataset(Dataset):
    def __init__(self, sequences, permutations, question):
        raise("This dataset has not yet been updated to support masked attention training")
        # use gpu for processing
        if type(sequences) == sparse._csr.csr_matrix:
          sequences = sequences.todense()

        data = []
        targets = []

        # generate the input output pairs
        # the target must be 0 or 1
        for sequence, permutation in tqdm(zip(sequences, permutations), desc="Loading data"):
          # shift the permutation to use the correct tokens
          # we don't want overlap between the input tokens and the output tokens
          shifted_perm = convert_perm_to_tokens(permutation)

          target = question(permutation)

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
            targets.append(target)

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

def load_data(dataset_class=MaskedDataset, question=None, skip_train=False, verbose=False):
  accelerator = Accelerator()
  should_speak = verbose and accelerator.is_local_main_process

  train_inputs = np.array([[0 for x in range(INPUT_LENGTH)]])
  train_perms = np.array([[0 for x in range(MAX_GROUP_SIZE)]])

  if not skip_train:
    if should_speak:
     print("Loading training data...")
    
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
  else:
    train_inputs = None
    train_perms = None
    dataset_size = None

  if should_speak:
     print("Loading validation data...")

  val_seqs = np.loadtxt(PATH + DATA + "val_data.csv", delimiter=",").astype(int)
  val_perms = np.loadtxt(PATH + DATA + "val_data_perms.csv", delimiter=",").astype(int)

  if should_speak:
     print("Loading test data...")

  test_seqs = np.loadtxt(PATH + DATA + "test_data.csv", delimiter=",").astype(int)
  test_perms = np.loadtxt(PATH + DATA + "test_data_perms.csv", delimiter=",").astype(int)

  # create the dataloaders
  if not skip_train:
    train_dataset = dataset_class(train_inputs, train_perms, question=question, mainthread=should_speak)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, num_workers=N_WORKERS)
  else:
    train_dataset = None
    train_dataloader = None

  val_dataset = dataset_class(val_seqs, val_perms, question=question, mainthread=should_speak)
  val_dataloader = DataLoader(val_dataset, batch_size=BATCHSIZE, num_workers=N_WORKERS)

  test_dataset = dataset_class(test_seqs, test_perms, question=question, mainthread=should_speak)
  test_dataloader = DataLoader(test_dataset, batch_size=BATCHSIZE, num_workers=N_WORKERS)

  return (
     train_inputs, train_perms, train_dataloader, 
     val_seqs, val_perms, val_dataloader,
     test_seqs, test_perms, test_dataloader,
     dataset_size
  )
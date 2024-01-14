from utilities import *
from config import *
from torch.utils.data import DataLoader, Dataset
from torch import tensor, float32, cuda, device
from scipy import sparse

class SimpleDataset(Dataset):
    def __init__(self, data, targets):
        # use gpu for processing
        if cuda.is_available():
          dev = "cuda:0"
        else:
          dev = "cpu"

        dev = device(dev)

        if type(data) == sparse._csr.csr_matrix:
          data = data.todense()

        self.data = tensor(data, dtype=int).to(dev)
        self.targets = tensor(targets.reshape(-1, 1), dtype=float32).to(dev)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = (
            self.data[index],
            self.targets[index]
        )
        return sample
    
class ExhaustiveDataset(Dataset):
    def __init__(self, calc_identity=True):
        # use gpu for processing
        if cuda.is_available():
          dev = "cuda:0"
        else:
          dev = "cpu"

        self.dev = device(dev)
        self.calc_identity = calc_identity

    def __len__(self):
        return GROUP_SIZE ** MAX_LENGTH

    def __getitem__(self, index):
        seq = int_to_seq(index)
        target = is_identity(seq) if self.calc_identity else 0

        sample = (
            tensor(seq, dtype=int).to(self.dev),
            tensor(target, dtype=float32).to(self.dev)
        )
        return sample

# load the data
print("Loading data...")
filenames = [f"train_data{x}" for x in range(1, TRAIN_FILES + 1)]

train_data = np.array([[0 for x in range(MAX_LENGTH)]])

for filename in filenames:
  data = np.loadtxt(PATH + DATA + filename + ".csv", delimiter=",").astype(int)
  train_data = np.concatenate((train_data, data))

train_data = train_data[1:]

val_data = np.loadtxt(PATH + DATA + "val_data.csv", delimiter=",").astype(int)
art_test_data = np.loadtxt(PATH + DATA + "art_test_data.csv", delimiter=",").astype(int)
true_test_data = np.loadtxt(PATH + DATA + "true_test_data.csv", delimiter=",").astype(int)

# only use first million of the true_test_data for ram concerns
true_test_data = true_test_data[:1000000]

# get number of identities in true_test_data
# print("True identity count:", len(true_test_data[np.apply_along_axis(is_identity, 1, true_test_data)]))

# label the datasets
X_train = train_data
X_val = val_data
X_art_test = art_test_data
X_true_test = true_test_data

y_train = np.apply_along_axis(is_identity, 1, X_train)
y_val = np.apply_along_axis(is_identity, 1, X_val)
y_art_test = np.apply_along_axis(is_identity, 1, X_art_test)
y_true_test = np.apply_along_axis(is_identity, 1, X_true_test)

final_dimension = MAX_LENGTH

# create the dataloaders
train_dataset = SimpleDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

val_dataset = SimpleDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

art_test_dataset = SimpleDataset(X_art_test, y_art_test)
art_test_dataloader = DataLoader(art_test_dataset, batch_size=64, shuffle=False, num_workers=0)

true_test_dataset = SimpleDataset(X_true_test, y_true_test)
true_test_dataloader = DataLoader(true_test_dataset, batch_size=64, shuffle=False, num_workers=0)
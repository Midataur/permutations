from utilities import *
from config import *

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
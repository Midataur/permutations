print("Loading libraries...")

import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from utilities import *
from config import *
from tqdm import tqdm
from transformer import *
import os

print("Logging in...")
wandb.login()

# this is the training script, assumes you're using the transformer
# if you're using the MLP, you'll need to change the data pipeline and the final dimension
# also you can modify the transformer config in the transformer.py file

print("Loading data...")

# load the data
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

# set hyperparameters
# some of these are in the transformer.py file

learning_rate = 0.00003
num_epochs = 10000

lr_factor = 0.1  # Factor by which the learning rate will be reduced
lr_patience = 10  # Number of epochs with no improvement after which learning rate will be reduced
threshold = 0.01  # Threshold for measuring the new optimum

# setup the model
model = BigramLanguageModel()

# cuda? (gpu)
if torch.cuda.is_available():
  device = "cuda:0"
else:
  device = "cpu"

# optionally: load the model
filename = PATH + "/model/" + MODELNAME + ".pth"
if os.path.isfile(filename):
    model.load_state_dict(torch.load(filename, map_location=torch.device(device)))

# send to gpu (maybe)
model = model.to(device)

# Define the loss function
criterion = nn.BCELoss()

# Define the optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=lr_factor,
    patience=lr_patience,
    threshold=threshold
)

print("Training...")

# train the model
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="permutations-binary-classifier",

    # track run hyperparameters and metadata
    config={
      "learning_rate": learning_rate,
      "scheduling": "plateau",
      "lr_factor": lr_factor,
      "lr_patience": lr_patience,
      "threshold": threshold,
      "architecture": "Transformer",
      "epochs": num_epochs,
      "optimizer": "AdamW",
      "identity_proportion": IDENTITY_PROPORTION,
      "group_size": GROUP_SIZE,
      "dataset_size": DATASET_SIZE,
      "max_length": MAX_LENGTH,
      "n_embed": n_embed,
      "n_head": n_head,
      "n_blocks": n_blocks,
      "dropout": dropout
    },
    settings=wandb.Settings(start_method="fork"),
    resume="allow",
    id=MODELNAME #CHECK IF THIS IS CORRECT
)

patience = 45
cur_patience = 0
best_loss = float("inf")

# training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    total_train_loss = 0.0
    total_train_accuracy = 0.0
    num_batches = 0

    print("Training...")
    for inputs, targets in tqdm(train_dataloader):  # Assuming you have a DataLoader for your training data
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Calculate the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        # stat track
        total_train_loss += loss.item()
        accuracy = calculate_accuracy(outputs, targets)
        total_train_accuracy += accuracy
        num_batches += 1

    average_train_accuracy = total_train_accuracy / num_batches
    train_loss = total_train_loss / num_batches

    # Calculate and print accuracy after each epoch
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode

        # calculate validation stats
        total_accuracy = 0.0
        total_loss = 0.0

        num_batches = 0

        print("Evaluating...")
        for inputs, targets in tqdm(val_dataloader):
            outputs = model(inputs)

            # calculate the val accuracy
            accuracy = calculate_accuracy(outputs, targets)
            total_accuracy += accuracy

            # Calculate the val loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1

        average_accuracy = total_accuracy / num_batches
        val_loss = total_loss / num_batches

        print(f"Epoch {epoch + 1}, Train loss {train_loss} Train Accuracy {average_train_accuracy} Validation Accuracy: {average_accuracy}, Val loss: {val_loss}")

        # log metrics to wandb
        wandb.log({
            "validation_accuracy": average_accuracy,
            "loss": val_loss,
            "training_accuracy": average_train_accuracy,
            "training_loss": train_loss,
        })

    # early stopping
    # if train_loss < best_loss:
    #     best_loss = train_loss
    #     cur_patience = 0
    #     torch.save(model.state_dict(), filename)
    # else:
    #     cur_patience += 1
        
    # always save the model
    torch.save(model.state_dict(), filename)

    # if cur_patience == patience:
    #     print("Early stopping activated")
    #     break

    if cur_patience % lr_patience == 0 and cur_patience != 0:
        print("Reducing learning rate")

    # learning rate scheduling
    scheduler.step(train_loss)
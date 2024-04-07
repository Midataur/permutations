print("Loading libraries...")

import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import wandb
from utilities import *
from config import *
from dataloading import *
from tqdm import tqdm
from transformer import *
import os

print("Logging in...")
wandb.login()

# this is the training script, assumes you're using the transformer
# if you're using the MLP, you'll need to change the data pipeline and the final dimension
# also you can modify the transformer config in the transformer.py file

# setup the model
model = BigramLanguageModel()

# cuda? (gpu)
if torch.cuda.is_available():
  device = "cuda:0"
else:
  device = "cpu"

# send to gpu (maybe)
model = nn.DataParallel(model)
model = model.to(device)

# optionally: load the model
filename = PATH + "/model/" + MODELNAME + ".pth"
if os.path.isfile(filename):
    model.load_state_dict(torch.load(filename, map_location=torch.device(device)))

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer and scheduler
optimizer = optim.AdamW(
    model.parameters(), 
    lr=learning_rate,
    weight_decay=weight_decay
)

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
    project="scaling-generator",

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
      "max_group_size": MAX_GROUP_SIZE,
      "dataset_size": DATASET_SIZE,
      "max_input_length": MAX_INPUT_LENGTH,
      "n_embed": n_embed,
      "n_head": n_head,
      "n_blocks": n_blocks,
      "dropout": dropout,
      "weight_decay": weight_decay,
      "base": TRANS_BASE,
      "batch_size": BATCHSIZE,
      "context_length": CONTEXT_LENGTH,
    },
    settings=wandb.Settings(start_method="fork"),
    resume="allow",
    id=MODELNAME #CHECK IF THIS IS CORRECT
)

patience = 45
cur_patience = 0
best_loss = float("inf")

last_train_loss = None
last_val_loss = None

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

        metrics = {
            "validation_accuracy": average_accuracy,
            "loss": val_loss,
            "training_accuracy": average_train_accuracy,
            "training_loss": train_loss,
        }

        # to show how fast we're plateauing
        if epoch > 0:
            metrics["delta_train_loss"] = train_loss - last_train_loss
            metrics["delta_val_loss"] = val_loss - last_val_loss
        
        last_train_loss = train_loss
        last_val_loss = val_loss

        print(f"Epoch {epoch + 1}, Train loss {train_loss} Train Accuracy {average_train_accuracy} Validation Accuracy: {average_accuracy}, Val loss: {val_loss}")

        # log metrics to wandb
        wandb.log(metrics)

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

print("done!", num_epochs)
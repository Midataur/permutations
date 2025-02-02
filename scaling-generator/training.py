import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import wandb
import config
from utilities import *
from dataloading import *
from tqdm.auto import tqdm
from transformer import *
from accelerate import Accelerator, load_checkpoint_and_dispatch
import os

def train(
        model_class=Transformer, 
        dataset_class=SimpleDataset, 
        question=None,
        suffix="",
        stop_block=None
    ):
    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        print("Logging in...")
        wandb.login()

        # this is the training script, assumes you're using the transformer
        # if you're using the MLP, you'll need to change the data pipeline and the final dimension
        # also you can modify the transformer config in the transformer.py file

        # load the data
        print("Loading data...")

    # i know this code is kinda bad, it's the result of tech debt
    (
        train_inputs, train_perms, train_dataloader, 
        val_seqs, val_perms, val_dataloader,
        test_seqs, test_perms, test_dataloader,
        dataset_size
    ) = load_data(dataset_class, question)

    # setup the model
    model = model_class(stop_block)

    # optionally: load the model
    save_directory = f"{PATH}/model/{MODELNAME}"
    file_path = f"{save_directory}/model.safetensors"
    
    if os.path.isfile(file_path):
        model = load_checkpoint_and_dispatch(model, file_path)

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

    # set up accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    val_dataloader = accelerator.prepare(val_dataloader)

    if accelerator.is_local_main_process:
        print("Training...")
        
        # train the model
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=f"scaling-generator{suffix}",

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
                "dataset_size": dataset_size,
                "input_length": INPUT_LENGTH,
                "n_embed": n_embed,
                "n_head": n_head,
                "n_blocks": n_blocks,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "batch_size": BATCHSIZE,
                "context_length": CONTEXT_LENGTH,
                "actual_group_size": ACTUAL_GROUP_SIZE,
                "window": WINDOW,
                "input_type": INPUT_TYPE,
                "legacy_override": LEGACY_OVERRIDE,
                "relabel": RELABEL,
                "floating_point_type": "bf16",
                "deepspeed": "enabled",
                "dataload_workers": N_WORKERS,
                "stop_block": stop_block,
                "window_count": WINDOW_COUNT,
                "partitioned_windows": PARTITIONED_WINDOWS
            },
            settings=wandb.Settings(start_method="fork"),
            resume="allow",
            id=f"{MODELNAME}{suffix}"
        )

    # patience = 45
    # cur_patience = 0
    # best_loss = float("inf")

    last_train_loss = None
    last_val_loss = None

    # training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        total_train_loss = 0.0
        total_train_accuracy = 0.0
        num_batches = 0

        if accelerator.is_local_main_process:
            print("Training...")
        
        for inputs, targets in tqdm(train_dataloader, disable=not accelerator.is_local_main_process):
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Calculate the loss
            accelerator.backward(loss)  # Backward pass
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

            if accelerator.is_local_main_process:
                print("Evaluating...")

            for inputs, targets in tqdm(val_dataloader, disable=not accelerator.is_local_main_process):
                outputs = model(inputs)

                all_outputs, all_targets = accelerator.gather_for_metrics((outputs, targets))

                # calculate the val accuracy
                accuracy = calculate_accuracy(outputs, targets)
                total_accuracy += accuracy

                # Calculate the val loss
                loss = criterion(all_outputs, all_targets)
                total_loss += loss.item()
                num_batches += 1

            average_accuracy = total_accuracy / num_batches
            val_loss = total_loss / num_batches

            metrics = {
                "validation_accuracy": average_accuracy,
                "loss": val_loss,
                "training_accuracy": average_train_accuracy,
                "training_loss": train_loss
            }

            # to show how fast we're plateauing
            if epoch > 0:
                metrics["delta_train_loss"] = train_loss - last_train_loss
                metrics["delta_val_loss"] = val_loss - last_val_loss
            
            last_train_loss = train_loss
            last_val_loss = val_loss

            if accelerator.is_local_main_process:
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
        accelerator.wait_for_everyone()
        accelerator.save_model(model, f"{save_directory}{suffix}")
        
        # save embedding pictures so we can make gifs later
        # this is broken since we added accelerate
        # TODO: FIX this later
        # if accelerator.is_local_main_process:
        #     save_embedding_pictures(model)

        # if cur_patience == patience:
        #     print("Early stopping activated")
        #     break

        # learning rate scheduling
        scheduler.step(train_loss)
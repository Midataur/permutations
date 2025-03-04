from utilities import *
from config import *
from dataloading import *
from transformer import *
from tqdm import tqdm
from accelerate import load_checkpoint_and_dispatch, Accelerator

def test():
    accelerator = Accelerator()

    should_talk = accelerator.is_main_process

    # i know this code is kinda bad, it's the result of tech debt
    (
        train_inputs, train_perms, train_dataloader, 
        val_seqs, val_perms, val_dataloader,
        test_seqs, test_perms, test_dataloader,
        dataset_size
    ) = load_data(skip_train=True, verbose=True)

    # setup the model
    model = Transformer()

    # optionally: load the model
    save_directory = f"{PATH}/model/{MODELNAME}"
    file_path = f"{save_directory}/model.safetensors"
    
    if os.path.isfile(file_path):
        if should_talk:
            print("Loaded the model!")
        model = load_checkpoint_and_dispatch(model, file_path)
    elif should_talk:
        print("Failed to load the model, defaulting to untrained model")

    model = accelerator.prepare(model)

    # check probability of getting it correct by default
    
    free_wins = 0
    for perm in test_perms:
        unstable = 0

        for pos, x in enumerate(perm):
            unstable += pos != x

        free_wins += unstable <= ACTUAL_GROUP_SIZE

    if should_talk:
        print("Model name:", MODELNAME)
        print("Free probability:", free_wins/len(test_perms))

    # test for all sequences
    results = []

    generate_function = model.generate if hasattr(model, "generate") else model.module.generate

    for seq, real_perm in (pbar:=tqdm(
        zip(test_seqs, test_perms), desc="Testing", total=len(test_perms), disable=not should_talk
    )):
        gen_perm = tuple(generate_function(seq, accelerator, force_valid=True))

        results.append((real_perm == gen_perm).all())
        pbar.set_description(f"Cur accuracy: {sum(results) / len(results)}")

    if should_talk:
        print(f"Accuracy: {sum(results) / len(results)}")

    # write results to a file that r can read
    if should_talk:
        print("Writing results to file")
    
    with open(PATH + "/results/" + MODELNAME + ".csv", "w") as f:
        f.write("results\n")
        for r in results:
            f.write(f"{r}\n")
        f.write("\n")

if __name__ == "__main__":
    test()
from utilities import *
from config import *
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import time

dataset_size = 1000000

# generates a matrix filled with random integers in [0, GROUP_SIZE-1]
# we say that 0 denotes the identity element
# words are uniformly distributed
def generate_random_sequences(num_seqs, max_length):
  # we create a new generator every time for parallel processing reasons
  generator = np.random.default_rng(seed=time.time_ns())

  return generator.integers(0, GROUP_SIZE, size=(num_seqs, max_length))

# generates a batch of identities
# this has nothing to do with pytorch batches
def make_batch(args):
  max_length, batch_size = args

  # generate some random sequences
  new_batch = generate_random_sequences(batch_size, max_length)

  # keep only the identities
  return new_batch[np.apply_along_axis(is_identity, 1, new_batch)]

# generates identity equivalents
def generate_identities(amount, max_length, batch_size=None,  workers=2, suppress=False):
  # default batch_size is amount * 4
  batch_size = batch_size if batch_size else amount

  # generate the identities
  identities = np.empty(shape=(1, max_length))
  generated = 0

  # go until we've generated enough
  with tqdm(total=amount, disable=suppress) as pbar:
    pbar.set_description("Generating identities")

    while generated < amount:
      # do some parallel processing
      with Pool() as pool:
        results = pool.map(make_batch, [(max_length, batch_size)]*workers)

        for batch in results:
          identities = np.concatenate((identities, batch))
          generated += len(batch)
          pbar.update(len(batch))

  # only keep the amount we need
  # the first one was a dummy, discard it
  identities = identities[1:amount+1]

  return np.array(identities, dtype=int)

if __name__ == "__main__":
    # generate the training data
    train_data = generate_random_sequences(
        int(
            dataset_size*(1-IDENTITY_PROPORTION)
        ),
        MAX_LENGTH
    )

    if IDENTITY_PROPORTION:
        train_data = np.concatenate(
            (
            train_data,
            generate_identities(
                int(
                dataset_size*IDENTITY_PROPORTION
                ),
                MAX_LENGTH, workers=1
            )
            )
        )

    print("True identity count:", len(train_data[np.apply_along_axis(is_identity, 1, train_data)]))

    # generate the validation data
    # the validation data has the same makeup as the training data
    # ie. it contains artifical identities

    val_data = generate_random_sequences(
        int(
            dataset_size*0.2*(1-IDENTITY_PROPORTION)
        ),
        MAX_LENGTH
    )

    if IDENTITY_PROPORTION:
        val_data = np.concatenate((
            val_data,
            generate_identities(
                int(
                dataset_size*0.2*IDENTITY_PROPORTION
                ),
                MAX_LENGTH
            )
        ))

    print("True identity count:", len(val_data[np.apply_along_axis(is_identity, 1, val_data)]))

    # generate the artificial test data
    # the artificial test data has the same makeup as the training data
    # ie. it does contain artifical identities

    art_test_data = generate_random_sequences(
        int(
            dataset_size*0.2*(1-IDENTITY_PROPORTION)
        ),
        MAX_LENGTH
    )

    if IDENTITY_PROPORTION:
        art_test_data = np.concatenate((
            val_data,
            generate_identities(
                int(
                dataset_size*0.2*IDENTITY_PROPORTION
                ),
                MAX_LENGTH
            )
        ))
        
    print("True identity count:", len(art_test_data[np.apply_along_axis(is_identity, 1, art_test_data)]))

    # generate the true test data
    # the test data is completely random
    # ie. it does not contain artifical identities

    true_test_data = generate_random_sequences(
        int(
            dataset_size
        ),
        MAX_LENGTH
    )

    print("True identity count:", len(true_test_data[np.apply_along_axis(is_identity, 1, true_test_data)]))

    # prompt: pickle the 4 datasets
    np.savetxt(PATH + "train_data.csv", train_data, delimiter=",")
    np.savetxt(PATH + "val_data.csv", val_data, delimiter=",")
    np.savetxt(PATH + "art_test_data.csv", art_test_data, delimiter=",")
    np.savetxt(PATH + "true_test_data.csv", true_test_data, delimiter=",")
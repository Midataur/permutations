from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from torch import argmax
from accelerate import Accelerator
import numpy as np
from config import *
import os

# moves all the zeroes to the end
class ZeroShifter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
      X = np.copy(X)

      for pos, sequence in enumerate(X):
        new_sequence = [0 for x in range(len(X[0]))]

        counter = 0
        for word in sequence:
          if word != 0:
            new_sequence[counter] = word
            counter += 1

        X[pos] = new_sequence

      return X

# reduce the dimensionality of the dataset without losing information
# this makes the training process faster and the resulting model is just as good
# we do lose the ability to interpret what features are useful unfortunately    
# we don't actually use this anymore
class DimReducer(BaseEstimator, TransformerMixin):

    def __init__(self, ratio=0.999999999, suppress=False):
        self.ratio = ratio
        self.suppress = suppress
        self.svd = None

    def fit(self, X, y=None):
        # do a principle component analysis
        old_dim = X.shape[1]
        self.svd = TruncatedSVD(n_components=old_dim)
        self.svd.fit(X)

        # find the minimum dimensionality that explains the required variance
        for pos, x in enumerate(self.svd.explained_variance_ratio_.cumsum()):
            if x >= self.ratio:
                break

        # get the real dimensionality
        real_dim = pos

        self.svd = TruncatedSVD(n_components=real_dim)
        self.svd.fit(X)

        if not self.suppress:
            print(f"Reduced dimensionality from {old_dim} to {real_dim}")
            reduction_factor = (old_dim-real_dim)/(old_dim/100)
            print(f"Reduction factor: {round(reduction_factor, 2)}%")

        return self

    def transform(self, X):
        if self.svd == None:
            raise Exception("You must call fit() on the data before calling transform()")

        # reduce the dimensionality
        return self.svd.transform(X)

def calculate_accuracy(output, target):
    # targets is a (B) tensor of integers that have the index of the correct class
    # we need to see if the max logit is at the right index
    return (argmax(output, dim=1) == target).float().mean()

# takes a permutation and converts it to tokens
def convert_perm_to_tokens(perm):
    return [int(char + num_trans) for char in perm]

def convert_tokens_to_perm(tokens):
    return [int(token - num_trans) for token in tokens]

# takes a token and tells you what type it is
def token_type(token):
    if token < num_trans:
        return "transposition"
    elif token < num_normal:
        return "permutation"
    return "special"

# saves the embedding similarity matrices so we can make a gif later
def save_embedding_pictures(model):
    posindices, tokindices = (
        torch.arange(block_size).cpu(),
        torch.arange(vocab_size).cpu()
    )

    if "module" in dir(model):
        model = model.module

    types = [
        ("position", model.position_embedding.detach().cpu()(posindices)),
        ("token", model.token_embedding_table.detach().cpu()(tokindices))
    ]

    for embedding_type, embedding in types:
        # generate the picture
        embedding = embedding.detach().cpu().numpy()

        similarity = []

        for x in embedding:
            row = []
            for y in embedding:
                row.append(np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y)))
            similarity.append(row)

        # save the picture
        # this is a fancy way to get the array to have shape (1, x, x)
        pictures = np.array(similarity).transpose()[:,:,np.newaxis].transpose()

        filename = f"./embedding_pictures/{embedding_type}/{MODELNAME}.npy"

        if os.path.isfile(filename):
            with open(filename, "rb") as file:
                old_pictures = np.load(file)
                pictures = np.concatenate((old_pictures, pictures))

        with open(filename, "wb") as file:
            np.save(file, pictures)

def reshape_outputs(outputs, targets):
    B, T, C = outputs.shape

    if MASKED_MODEL:
        return outputs.reshape(B*T, C), targets.reshape(B*T)
    
    return outputs, targets
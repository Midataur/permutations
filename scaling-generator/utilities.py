from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from torch import argmax
import numpy as np
from config import *

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
  return [char + TRANS_BASE for char in perm]

def convert_tokens_to_perm(tokens):
  return [token - TRANS_BASE for token in tokens]
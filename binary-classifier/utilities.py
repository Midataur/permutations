from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
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

# applies S_i in place
def apply_word(perm, i):
  # 0 is the identity
  if i:
    if TRANSPOSITION_TYPE == "general":
      x = i // GROUP_SIZE
      y = i % GROUP_SIZE

      perm[x], perm[y] = perm[y], perm[x]
    elif TRANSPOSITION_TYPE == "elementary":
      perm[i-1], perm[i] = perm[i], perm[i-1]

def get_permutation(sequence):
  permutation = list(range(GROUP_SIZE))

  for word in sequence:
    apply_word(permutation, word)

  return permutation

# used to exhaustively generate all possible sequences
def int_to_seq(num):
  sequence = []

  for i in range(MAX_LENGTH):
    sequence.append(num % GROUP_SIZE)
    num //= GROUP_SIZE

  return sequence

# tells you if a sequence corresponds to the identity
# we need this because there is a non_zero chance that we generate one randomly
# probably wouldn't be the end of the world (low chance this happens)
# but better safe than sorry
def is_identity(sequence):
  return get_permutation(sequence) == list(range(GROUP_SIZE))

# Example of accuracy calculation (you may need to adjust it based on your specific requirements)
def calculate_accuracy(output, target):
    return (output.round() == target).float().mean()
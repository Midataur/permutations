from itertools import permutations
from tqdm import tqdm
from sympy import Rational, simplify, symbols
from sympy.matrices.sparse import SparseMatrix
import numpy as np
import pyperclip

GROUP_SIZE = 6

def swap(perm, index):
    perm[index], perm[index+1], = perm[index+1], perm[index]

adj_matrix = []
all_perms = list(permutations(range(GROUP_SIZE), GROUP_SIZE))

print("Computing adjacency matrix...")

for perm in tqdm(all_perms):
    row = [0 for x in range(len(all_perms))]
    row[all_perms.index(perm)] = 1

    for x in range(GROUP_SIZE-1):
        test = list(perm)
        swap(test, x)
        row[all_perms.index(tuple(test))] = 1

    adj_matrix.append(row)

adj_matrix = np.array(adj_matrix)
trans_matrix = SparseMatrix([
    [Rational(x) for x in row] for row in adj_matrix
])/GROUP_SIZE

pyperclip.copy(str(trans_matrix*GROUP_SIZE).replace("[","{").replace("]","}")[7:-1])
print("Adjacency matrix copied")

print("Computing diagonalization...")
P, D = trans_matrix.diagonalize()
n = symbols("n")
print("Computing power...")
expr = (P*(D**n)*P.inv())
print("Simplifying expression...")
print(simplify(expr))
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, vstack, hstack, save_npz, load_npz, block_diag, identity, random
from scipy.sparse.linalg import inv, spsolve, splu
from scipy.linalg import lu, lu_factor, lu_solve
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, shared_memory
import matplotlib.pyplot as plt
import time
import cProfile
import pstats 
import csv
import os 
from tqdm import tqdm
import gdown

def lower_block_bidiagonal_nonsingular(n_blocks, block_size):
    """
    Generate a nonsingular sparse lower block bidiagonal matrix in CSR format.

    Parameters:
        n_blocks (int): Number of diagonal blocks.
        block_size (int): Size of each square block.

    Returns:
        scipy.sparse.csr_matrix: The resulting nonsingular sparse matrix.
        numpy.ndarray: The corresponding RHS vector.
    """
    N = n_blocks * block_size  # Total size of the matrix
    data, row_indices, col_indices = [], [], []

    # Generate diagonal (B_i) and lower diagonal (L_i) blocks
    for i in range(n_blocks):
        row_offset = i * block_size
        col_offset = i * block_size

        # Ensure nonzero entries in the main diagonal block (B_i)
        block_main = np.random.rand(block_size, block_size) + np.eye(block_size)  # Make B_i non-singular
        for r in range(block_size):
            for c in range(block_size):
                val = block_main[r, c]
                data.append(val)
                row_indices.append(row_offset + r)
                col_indices.append(col_offset + c)

        # Lower block (L_i), ensuring nonzero entries
        if i < n_blocks - 1:
            row_offset = (i + 1) * block_size
            col_offset = i * block_size
            block_lower = np.random.rand(block_size, block_size) + 0.5*np.eye(block_size) # Random values ensure nonzero entries

            for r in range(block_size):
                for c in range(block_size):
                    val = block_lower[r, c]
                    data.append(val)
                    row_indices.append(row_offset + r)
                    col_indices.append(col_offset + c)

    # Create sparse CSR matrix
    sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(N, N))

    # Generate a random RHS vector (column vector)
    rhs_vector = np.random.rand(N, 1)  # Nx1 dense vector

    return sparse_matrix, rhs_vector

block_size = 4
number_of_processors = 4

#k_list = [4,5,6,7,8,9,10,11,12,13,14,15,16]
k_list = [18]
for k in tqdm(k_list):
    n = int(number_of_processors*2**k)
    number_of_blocks = n + 1
    print(k)
    print(number_of_blocks)
    M, f = lower_block_bidiagonal_nonsingular(number_of_blocks, block_size)
    x = spsolve(M,f)
    save_folder = f"Samples_to_test"
    save_npz(f"n{number_of_blocks}_b{block_size}_mat.npz",M)
    np.save(f"n{number_of_blocks}_b{block_size}_rhs.npy",f)
    np.save(f"n{number_of_blocks}_b{block_size}_sol.npy",x)
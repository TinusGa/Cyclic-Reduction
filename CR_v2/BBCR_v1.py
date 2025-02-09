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

def create_full_permutation_matrix(m, block_size):
    num_blocks = m // block_size
    perm_order = []

    # First, append all odd-indexed blocks
    for i in range(0, num_blocks, 2):
        perm_order.append(i)

    # Then, append all even-indexed blocks
    for i in range(1, num_blocks, 2):
        perm_order.append(i)

    data, rows, cols = [], [], []

    for new_index, old_index in enumerate(perm_order):
        start_new = new_index * block_size
        start_old = old_index * block_size
        for i in range(block_size):
            data.append(1)              
            rows.append(start_new + i)  
            cols.append(start_old + i)  

    return csr_matrix((data, (rows, cols)), shape=(m, m))

def placeholder_name(M, f, block_size : int, processors : int):
    """ 
    Performs Block Cyclic Reduction (BCR) in parallel for solving lower block bidiagonal systems.

    Parameters:
    -----------
    M : scipy.sparse.csr_matrix or numpy.ndarray
        The coefficient matrix of size (N, N), where N = (n+1) * block_size.
        Must be a square lower block bidiagonal matrix.

    f : numpy.ndarray
        The right-hand side (RHS) vector of size (N, 1), corresponding to Mx = f.

    block_size : int
        The size of each block in the matrix.

    processors : int
        The number of processors used for parallel block cyclic reduction.

    Returns:
    --------
    x : numpy.ndarray
        The solution vector of size (N, 1) satisfying Mx = f.
    """
    N, L = M.shape
    assert N == L,  f"M must be sqaure but has dimensions {N}x{L}"
    n = (N - 1) // block_size
    assert n % processors == 0, f"M must have size (n+1)*block_size x (n+1)*block_size, where n = p * 2**k. p is not a multiple of n."
    nbyp = n // processors 
    assert ((nbyp & (nbyp-1) == 0) and nbyp != 0), f"M must have size (n+1)*block_size x (n+1)*block_size, where n = p * 2**k. n/p is not a power of two."
    number_of_steps = int(np.log2(nbyp)) # Number of steps in the forward step and backward step for each processor

    row_index_start = block_size
    row_index_end = block_size*(1+nbyp)
    col_index_start = 0
    col_index_end = block_size*(nbyp+1)
    
    # Divide among the processors
    M_k_list = []
    f_k_list = []
    B_k_s_list = []
    A_k_s_list = []
    f_k_s_list = []
    
    for _ in range(processors):
        # Perform the forward step
        M_copy = M[row_index_start:row_index_end, col_index_start:col_index_end]
        f_copy = f[row_index_start:row_index_end]
        M_k, f_k, B_k_s, A_k_s, f_k_s = forward_placeholder(M_copy, f_copy, block_size, processors, [], [], [])

        # Store the results for inter-processor communication
        M_k_list.append(M_k)
        f_k_list.append(f_k)

        # Store the results for the backward step
        B_k_s_list.append(B_k_s)
        A_k_s_list.append(A_k_s)
        f_k_s_list.append(f_k_s)

        # Update the indices for the next processor
        row_index_start = row_index_end
        row_index_end += nbyp*block_size 
        col_index_start = col_index_end - block_size
        col_index_end += block_size*nbyp
    
    x0 = spsolve(M[:block_size,:block_size],f[:block_size]) # The master of all processors
    base_case_x = [x0]

    # Only serial part of the algorithm
    for i in range(processors):
        B_k = M_k_list[i][:,:block_size]
        A_k = M_k_list[i][:,block_size:]
        f_k = f_k_list[i]
        x_next = spsolve(A_k,f_k.flatten()-B_k@base_case_x[i])
        base_case_x.append(x_next)
    
    final_x = np.array([])
    
    # Perform the backward step
    for i in range(processors):
        B_k_s = B_k_s_list[i] 
        A_k_s = A_k_s_list[i] 
        f_k_s = f_k_s_list[i] 

        x_for_current_processor = backward_placeholder(B_k_s, A_k_s, f_k_s, base_case_x[i:i+2], number_of_steps, block_size, processors)
        #x_for
        final_x = np.concatenate((final_x, x_for_current_processor))

    final_x = np.concatenate((final_x, base_case_x[-1]))
    return final_x
        
def forward_placeholder(M, f, block_size : int, processors : int, B_s = [], A_s = [], f_s = []):
    n,m = M.shape
    if n == block_size:
        return M,f,B_s,A_s,f_s
    
    M_next = csr_matrix((n//2,n//2+block_size))
    f_next = np.zeros(n//2)
    # Do one step
    for i in range(0,n,2*block_size):
        # Extract block elements from input
        B1 = M[i:i+block_size, i:i+block_size] 
        A1 = M[i:i+block_size, i + block_size: i + 2*block_size] 
        B2 = M[i + block_size: i + 2*block_size,i + block_size: i + 2*block_size ]
        A2 = M[i + block_size: i + 2*block_size,i + 2*block_size: i + 3*block_size]
        f1 = f[i:i+block_size]
        f2 = f[i + block_size: i + 2*block_size]

        # Store the values for the backward step
        B_s.append(B1)
        A_s.append(A1)
        f_s.append(f1)

        ####################
        # Convert blocks to dense arrays if they aren’t already.
        B1_dense = B1.toarray() if hasattr(B1, "toarray") else B1
        A1_dense = A1.toarray() if hasattr(A1, "toarray") else A1
        B2_dense = B2.toarray() if hasattr(B2, "toarray") else B2
        A2_dense = A2.toarray() if hasattr(A2, "toarray") else A2
        f1_dense = f1.toarray() if hasattr(f1, "toarray") else f1
        f2_dense = f2.toarray() if hasattr(f2, "toarray") else f2

        # Use LU factorization with pivoting to "solve" for the necessary inverses.
        lu_A1, piv_A1 = lu_factor(A1_dense)
        lu_B2, piv_B2 = lu_factor(B2_dense)

        # Instead of explicitly forming inverses, we solve linear systems:
        new_B1 = lu_solve((lu_A1, piv_A1), B1_dense)
        new_A1 = -lu_solve((lu_B2, piv_B2), A2_dense)
        new_f1 = lu_solve((lu_A1, piv_A1), f1_dense) - lu_solve((lu_B2, piv_B2), f2_dense)
        ####################

        # Compute inverses and values for the next depth. This is equivalent to removing all odd indices from the input
        # B2_inv = inv(B2)
        # A1_inv = inv(A1)
        # new_B1 = A1_inv@B1
        # new_A1 = -B2_inv@A2
        # new_f1 = A1_inv@f1 - B2_inv@f2

        # Set the new values to obtain a reduced system of half the original size
        j = i//2
        M_next[j:j+block_size,j:j+block_size] = new_B1
        M_next[j:j+block_size,j+block_size:j+2*block_size] = new_A1
        f_next[j:j+block_size] = new_f1.flatten()

    # Recursively apply the same procedure
    return forward_placeholder(M_next,f_next,block_size,processors,B_s,A_s,f_s)

def backward_placeholder(B_s, A_s, f_s, x_s, number_of_steps : int, block_size : int, processors : int):
    x_result = x_s[0]
    power = 0
    for i in range(number_of_steps-1,-1,-1):
        j = -(2**power - 1) if power > 0 else None
        k = -(2**(power+1) - 1)

        B = B_s[k:j]
        A = A_s[k:j]
        f = f_s[k:j]

        A_for_solve = block_diag(A, format='csr')
        B_for_solve = block_diag(B, format='csr')
        f_for_solve = np.concatenate(f).flatten()
        x_for_solve = x_result.copy()   

        x_new = spsolve(A_for_solve,f_for_solve - B_for_solve@x_for_solve)

        x_result = np.concatenate((x_result,x_new))
        Q = create_full_permutation_matrix(x_result.shape[0], block_size)
        x_result = Q.T@x_result
        power += 1
    return x_result
    
if __name__ == "__main__":
    block_size = 4
    number_of_processors = 8
    number_of_blocks_list = [33,65,129,257,513,1025,2049,4097,8193,16385,32769,65537,131073]
    number_of_blocks_list = [33]
    # Load pre-generated matrix and RHS vector
    for number_of_blocks in number_of_blocks_list:
        print(f"M.shape = {number_of_blocks*block_size}x{number_of_blocks*block_size}")
        save_folder = f"Samples_to_test"
        M,f,x = load_npz(f"{save_folder}/n{number_of_blocks}_b{block_size}_mat.npz"), np.load(f"{save_folder}/n{number_of_blocks}_b{block_size}_rhs.npy"), np.load(f"{save_folder}/n{number_of_blocks}_b{block_size}_sol.npy")
        start = time.time()
        x_sol = placeholder_name(M,f,block_size=block_size,processors=number_of_processors)
        end = time.time()
        print(f"Time taken: {end-start} seconds")
        print("Error,", np.linalg.norm(x - x_sol))

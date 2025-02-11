import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, vstack, hstack, save_npz, load_npz, block_diag, identity, random
from scipy.sparse.linalg import inv, spsolve, splu
from scipy.linalg import lu, lu_factor, lu_solve
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, shared_memory
from numba import njit
import matplotlib.pyplot as plt
import time
import cProfile
import pstats 
import csv
import os 
from tqdm import tqdm
import gdown

import_remote_test_files = True

if import_remote_test_files:
    url = "https://drive.google.com/drive/folders/1HWFHKCprFzR7H7TYhrE-W7v4bz2Vc7Ia"
    gdown.download_folder(url, quiet=True, use_cookies=False)

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

def BCR(M, f, block_size : int, processors : int):
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
    
    # Divide among the processors
    M_k_list = []
    f_k_list = []
    B_k_s_list = []
    A_k_s_list = []
    f_k_s_list = []

    for i in range(processors):
        start = time.time()
        # Perform the forward step
        M_list_copy = [] # [ [B1, A1], [B2, A2], ... ]
        f_list_copy = [] # [f1, f2, ...]
        start_row = block_size*(nbyp*i+1)
        end_row = block_size*(nbyp*(i+1)+1)
      
        rows = M[start_row:end_row].toarray()
        for j in range(start_row, end_row, block_size):
            rel = j - start_row  
            B1 = rows[rel:rel+block_size, j - block_size:j]
            A1 = rows[rel:rel+block_size, j:j+block_size]
            M_list_copy.append([B1, A1])
            f_list_copy.append(f[j:j+block_size])
        

        build_time += time.time() - start
        # Keep in case we need this later!
        # for j in range(start_row, end_row, block_size):
        #     row = M[j:j+block_size].toarray()
        #     M_list_copy.append([row[:,j-block_size:j], row[:,j:j+block_size]])
        # for j in range(start_row, end_row, block_size):
        #     f_list_copy.append(f[j:j+block_size])

        start = time.time()
        M_k, f_k, B_k_s, A_k_s, f_k_s = forward_reduction(M_list_copy, f_list_copy, block_size, processors)
        forward_time += time.time() - start
        # Store the results for inter-processor communication
        M_k_list.append(M_k)
        f_k_list.append(f_k)

        # Store the results for the backward step
        B_k_s_list.append(B_k_s)
        A_k_s_list.append(A_k_s)
        f_k_s_list.append(f_k_s)
    
    x0 = spsolve(M[:block_size,:block_size],f[:block_size]) # The master of all processors
    base_case_x = [x0]

    # Only serial part of the algorithm
    for i in range(processors):
        B_k = M_k_list[i][0]
        A_k = M_k_list[i][1]
        f_k = f_k_list[i]
        x_next = np.linalg.solve(A_k,f_k.flatten()-B_k@base_case_x[i])
        base_case_x.append(x_next)
    
    final_x = np.array([])

    # Perform the backward step
    for i in range(processors):
        B_k_s = B_k_s_list[i] 
        A_k_s = A_k_s_list[i] 
        f_k_s = f_k_s_list[i] 
        start = time.time()
        x_for_current_processor = backsubstitution(B_k_s, A_k_s, f_k_s, base_case_x[i:i+2], number_of_steps, block_size, processors)
        backward_time += time.time() - start
        final_x = np.concatenate((final_x, x_for_current_processor))

    final_x = np.concatenate((final_x, base_case_x[-1]))
    print(f"Build time: {build_time} seconds")  
    print(f"Forward time: {forward_time} seconds")
    print(f"Backward time: {backward_time} seconds")
    return final_x


#@njit
def forward_reduction(M, f, block_size: int, processors: int):
    """
    Performs the forward reduction step iteratively for block cyclic reduction,
    building the new system matrix by accumulating data rather than using slice assignments.

    Parameters:
    -----------
    M : scipy.sparse.csr_matrix
        The current block matrix.
    f : numpy.ndarray
        The current right-hand side vector.
    block_size : int
        The size of each block.
    processors : int
        The number of processors (unused in this iterative version, but kept for compatibility).
    B_s, A_s, f_s : list, optional
        Lists to store the blocks and right-hand sides needed for the backward step.
        
    Returns:
    --------
    M, f, B_s, A_s, f_s : tuple
        The reduced matrix and right-hand side, along with the lists needed for the backward step.
    """
    # Initialize storage lists if they are not provided.
    B_s = []
    A_s = []
    f_s = []

    # Continue reducing until the number of blocks m becomes 1.
    while True:
        m = len(M) # List now has size 1x2 i.e., [[B_k, A_k]]
        if m == 1:
            break

        num_new_blocks = m // 2
        # The new system will have:
        #   - rows: block_size * (m//2)
        #   - columns: block_size * ((m//2) + 1)
        n_rows = block_size * num_new_blocks
        n_cols = block_size * (num_new_blocks + 1)
        
        # Prepare lists to accumulate nonzero values and their row and column indices.
        data = []
        rows = []
        cols = []
        # Also, accumulate new right-hand side blocks.
        M_next_list = []
        f_next_list = []
        
        I = np.eye(block_size)
        
        # Process two blocks at a time.
        for i in range(0, m, 2):
            # Extract the blocks B1, A1, B2, A2, and f1, f2 in pairs of rows from M and f.

            row_1 = M[i]
            row_2 = M[i+1]
            B1 = row_1[0]
            A1 = row_1[1]
            B2 = row_2[0]
            A2 = row_2[1]

            f1 = f[i]
            f2 = f[i+1]

            # Save blocks needed for the backward step.
            B_s.append(B1)
            A_s.append(A1)
            f_s.append(f1)

            # Solve for the new block coefficients.
            A1_inv = np.linalg.solve(A1, I)
            #A1_inv = np.linalg.solve(A1, I)
            B2A1_inv = B2 @ A1_inv
            new_B1 = B2A1_inv @ B1
            new_A1 = -A2
            new_f1 = B2A1_inv @ f1 - f2

            M_next_list.append([new_B1, new_A1])

            # Accumulate the new right-hand side block.
            f_next_list.append(new_f1.flatten())

        # Update M and f for the next iteration.
        M, f = M_next_list, f_next_list

    return M[0], f[0], B_s, A_s, f_s


def backsubstitution(B_s, A_s, f_s, x_s, number_of_steps : int, block_size : int, processors : int):
    x_result = x_s[0]
    power = 0
    for i in range(number_of_steps-1,-1,-1):
        j = -(2**power - 1) if power > 0 else None
        k = -(2**(power+1) - 1)

        # Each of these is a list of blocks (dense arrays).
        B_blocks = B_s[k:j]
        A_blocks = A_s[k:j]
        f_blocks = f_s[k:j]

        # Determine how many blocks we need to solve in this step.
        num_step_blocks = len(A_blocks)
        if num_step_blocks == 0:
            break

        x_for_solve = x_result.copy()
        x_for_solve_blocks = [
            x_for_solve[i * block_size: (i + 1) * block_size]
            for i in range(num_step_blocks)
        ]

        # Solve each block: the block system is
        #     A_blocks[i] * x_new_block = f_blocks[i] - B_blocks[i] @ (x_for_solve corresponding block)
        x_new_list = []
        for idx in range(num_step_blocks):
            rhs = f_blocks[idx].flatten() - B_blocks[idx] @ x_for_solve_blocks[idx]
            x_new_i = np.linalg.solve(A_blocks[idx], rhs)
            x_new_list.append(x_new_i)
        x_new = np.concatenate(x_new_list)

        # Update x_result by concatenating the old solution with the newly computed blocks.
        x_result = np.concatenate((x_result, x_new))

        # Instead of building a permutation matrix, compute and apply permutation indices. Similar to Q.T @ x_result
        m_total = x_result.shape[0]
        num_blocks = m_total // block_size
        even_blocks = np.arange(0, num_blocks, 2)
        odd_blocks  = np.arange(1, num_blocks, 2)
        forward_perm = np.concatenate((even_blocks, odd_blocks))
        inv_perm = np.argsort(forward_perm)
        new_indices = np.repeat(inv_perm, block_size) * block_size + np.tile(np.arange(block_size), num_blocks)

        x_result = x_result[new_indices]

        power += 1
    return x_result
    
if __name__ == "__main__":
    block_size = 4
    number_of_processors = 4
    number_of_blocks_list = [33,65,129,257,513,1025,2049,4097,8193,16385,32769,65537,131073]
    number_of_blocks_list = [8193]
    # Load pre-generated matrix and RHS vector
    cprofiler = False

    if cprofiler:
        profiler = cProfile.Profile()
        profiler.enable()
        processes = [1]

    for number_of_blocks in number_of_blocks_list:
        print(f"M.shape = {number_of_blocks*block_size}x{number_of_blocks*block_size}")
        save_folder = f"BCR_v2"
        M,f,x = load_npz(f"{save_folder}/n{number_of_blocks}_b{block_size}_mat.npz"), np.load(f"{save_folder}/n{number_of_blocks}_b{block_size}_rhs.npy"), np.load(f"{save_folder}/n{number_of_blocks}_b{block_size}_sol.npy")
        start = time.time()
        x_sol = BCR(M,f,block_size=block_size,processors=number_of_processors)
        end = time.time()
        print(f"Time taken: {end-start} seconds")
        print("Error,", np.linalg.norm(x - x_sol))
    
    if cprofiler:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.sort_stats("time").print_stats(20)

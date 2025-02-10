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
    
    # Divide among the processors
    M_k_list = []
    f_k_list = []
    B_k_s_list = []
    A_k_s_list = []
    f_k_s_list = []
    
    for i in range(processors):
        # Perform the forward step
        M_copy = M[
            block_size*(nbyp*i+1):block_size*(nbyp*(i+1)+1),
            block_size*(nbyp*i):block_size*(nbyp*(i+1)+1)
        ]
        f_copy = f[
            block_size*(nbyp*i+1):block_size*(nbyp*(i+1)+1)
        ]
        M_k, f_k, B_k_s, A_k_s, f_k_s = forward_placeholder(M_copy, f_copy, block_size, processors, [], [], [])
        #print("One Forward complete ##############################")

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
        final_x = np.concatenate((final_x, x_for_current_processor))

    final_x = np.concatenate((final_x, base_case_x[-1]))
    return final_x
        
# def forward_placeholder(M, f, block_size : int, processors : int, B_s = [], A_s = [], f_s = []):
#     m = M.shape[0]//block_size
#     if m == 1:
#         return M,f,B_s,A_s,f_s
    
    
#     M_next = csr_matrix((block_size*m//2,block_size*m//2+block_size))
#     f_next = np.zeros(block_size*m//2)
#     I = np.eye(block_size)
#     # Do one step
#     for i in range(0,m,2):
#         # Extract block elements from input
#         B1 = M[
#             block_size*i:block_size*(i+1), 
#             block_size*i:block_size*(i+1)
#             ] 
#         A1 = M[
#             block_size*i:block_size*(i+1), 
#             block_size*(i+1):block_size*(i+2)
#             ] 
#         B2 = M[
#             block_size*(i+1):block_size*(i+2),
#             block_size*(i+1):block_size*(i+2)
#             ]
#         A2 = M[
#             block_size*(i+1):block_size*(i+2),
#             block_size*(i+2):block_size*(i+3)
#             ]
#         f1 = f[block_size*i:block_size*(i+1)]
#         f2 = f[block_size*(i+1):block_size*(i+2)]

#         # Store the values for the backward step
#         B_s.append(B1)
#         A_s.append(A1)
#         f_s.append(f1)

    
#         A1_inv = spsolve(A1, I)
#         B2A1_inv = B2@A1_inv
#         new_B1 = B2A1_inv@B1
#         new_A1 = -A2
#         new_f1 = B2A1_inv@f1 - f2
#         # Set the new values to obtain a reduced system of half the original size
#         j = i//2
#         M_next[block_size*j:block_size*(j+1),block_size*j:block_size*(j+1)] = new_B1
#         M_next[block_size*j:block_size*(j+1),block_size*(j+1):block_size*(j+2)] = new_A1
#         f_next[block_size*j:block_size*(j+1)] = new_f1.flatten()

#     # Recursively apply the same procedure
#     return forward_placeholder(M_next,f_next,block_size,processors,B_s,A_s,f_s)

# def forward_placeholder(M, f, block_size: int, processors: int, B_s=None, A_s=None, f_s=None):
#     """
#     Performs the forward reduction step iteratively (instead of recursively)
#     for block cyclic reduction.
    
#     Parameters:
#     -----------
#     M : scipy.sparse.csr_matrix
#         The current block matrix.
#     f : numpy.ndarray
#         The current right-hand side vector.
#     block_size : int
#         The size of each block.
#     processors : int
#         The number of processors (unused in this iterative version, but kept for compatibility).
#     B_s, A_s, f_s : list, optional
#         Lists to store the blocks and right-hand sides needed for the backward step.
        
#     Returns:
#     --------
#     M, f, B_s, A_s, f_s : tuple
#         The reduced matrix and right-hand side, along with the lists needed for the backward step.
#     """
#     # Initialize storage lists if they are not provided.
#     if B_s is None:
#         B_s = []
#     if A_s is None:
#         A_s = []
#     if f_s is None:
#         f_s = []

#     # Continue reducing until the number of blocks m becomes 1.
#     while True:
#         m = M.shape[0] // block_size
#         if m == 1:
#             break

#         num_new_blocks = m // 2
#         # The new system will have:
#         #   - rows: block_size * (m//2)
#         #   - columns: block_size * ((m//2) + 1)
#         M_next = csr_matrix((block_size * num_new_blocks, block_size * (num_new_blocks + 1)))
#         f_next = np.zeros(block_size * num_new_blocks)
#         I = np.eye(block_size)

#         # Process two blocks at a time.
#         for i in range(0, m, 2):
#             # Extract blocks from M and corresponding segments from f.
#             B1 = M[ block_size * i       : block_size * (i + 1),
#                     block_size * i       : block_size * (i + 1)]
#             A1 = M[ block_size * i       : block_size * (i + 1),
#                     block_size * (i + 1) : block_size * (i + 2)]
#             B2 = M[ block_size * (i + 1) : block_size * (i + 2),
#                     block_size * (i + 1) : block_size * (i + 2)]
#             A2 = M[ block_size * (i + 1) : block_size * (i + 2),
#                     block_size * (i + 2) : block_size * (i + 3)]
#             f1 = f[ block_size * i       : block_size * (i + 1)]
#             f2 = f[ block_size * (i + 1) : block_size * (i + 2)]

#             # Save blocks needed for the backward step.
#             B_s.append(B1)
#             A_s.append(A1)
#             f_s.append(f1)

#             # Solve for the new block coefficients.
#             A1_inv = spsolve(A1, I)
#             B2A1_inv = B2 @ A1_inv
#             new_B1 = B2A1_inv @ B1
#             new_A1 = -A2
#             new_f1 = B2A1_inv @ f1 - f2

#             j = i // 2  # New block index.
#             # Assign computed blocks into the new matrix.
#             M_next[ block_size * j       : block_size * (j + 1),
#                     block_size * j       : block_size * (j + 1)] = new_B1
#             M_next[ block_size * j       : block_size * (j + 1),
#                     block_size * (j + 1) : block_size * (j + 2)] = new_A1
#             f_next[ block_size * j : block_size * (j + 1)] = new_f1.flatten()

#         # Update M and f for the next iteration.
#         M, f = M_next, f_next

#     return M, f, B_s, A_s, f_s


def forward_placeholder(M, f, block_size: int, processors: int, B_s=None, A_s=None, f_s=None):
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
    if B_s is None:
        B_s = []
    if A_s is None:
        A_s = []
    if f_s is None:
        f_s = []

    # Continue reducing until the number of blocks m becomes 1.
    while True:
        m = M.shape[0] // block_size
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
        f_next_list = []
        
        I = np.eye(block_size)

        # Process two blocks at a time.
        for i in range(0, m, 2):
            # Extract blocks from M and corresponding segments from f.
            B1 = M[ block_size * i       : block_size * (i + 1),
                    block_size * i       : block_size * (i + 1)]
            A1 = M[ block_size * i       : block_size * (i + 1),
                    block_size * (i + 1) : block_size * (i + 2)]
            B2 = M[ block_size * (i + 1) : block_size * (i + 2),
                    block_size * (i + 1) : block_size * (i + 2)]
            A2 = M[ block_size * (i + 1) : block_size * (i + 2),
                    block_size * (i + 2) : block_size * (i + 3)]
            f1 = f[ block_size * i       : block_size * (i + 1)]
            f2 = f[ block_size * (i + 1) : block_size * (i + 2)]

            # Save blocks needed for the backward step.
            B_s.append(B1)
            A_s.append(A1)
            f_s.append(f1)

            # Solve for the new block coefficients.
            A1_inv = spsolve(A1, I)
            B2A1_inv = B2 @ A1_inv
            new_B1 = B2A1_inv @ B1
            new_A1 = -A2
            new_f1 = B2A1_inv @ f1 - f2

            j = i // 2  # New block index.

            # --- Accumulate entries for new_B1 ---
            # new_B1 occupies rows [block_size*j, block_size*(j+1))
            # and columns [block_size*j, block_size*(j+1))
            r_offset = block_size * j
            c_offset = block_size * j
            for r in range(block_size):
                for c in range(block_size):
                    val = new_B1[r, c]
                    # Only store nonzero entries.
                    if val != 0:
                        data.append(val)
                        rows.append(r_offset + r)
                        cols.append(c_offset + c)

            # --- Accumulate entries for new_A1 ---
            # new_A1 occupies rows [block_size*j, block_size*(j+1))
            # and columns [block_size*(j+1), block_size*(j+2))
            r_offset = block_size * j
            c_offset = block_size * (j + 1)
            for r in range(block_size):
                for c in range(block_size):
                    val = new_A1[r, c]
                    if val != 0:
                        data.append(val)
                        rows.append(r_offset + r)
                        cols.append(c_offset + c)

            # Accumulate the new right-hand side block.
            f_next_list.append(new_f1.flatten())

        # Build the new CSR matrix in one go.
        M_next = csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
        f_next = np.concatenate(f_next_list)

        # Update M and f for the next iteration.
        M, f = M_next, f_next

    return M, f, B_s, A_s, f_s


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
    number_of_blocks_list = [1025]
    # Load pre-generated matrix and RHS vector
    cprofiler = True

    if cprofiler:
        profiler = cProfile.Profile()
        profiler.enable()
        processes = [1]

    for number_of_blocks in number_of_blocks_list:
        print(f"M.shape = {number_of_blocks*block_size}x{number_of_blocks*block_size}")
        save_folder = f"Samples_to_test"
        M,f,x = load_npz(f"{save_folder}/n{number_of_blocks}_b{block_size}_mat.npz"), np.load(f"{save_folder}/n{number_of_blocks}_b{block_size}_rhs.npy"), np.load(f"{save_folder}/n{number_of_blocks}_b{block_size}_sol.npy")
        start = time.time()
        x_sol = placeholder_name(M,f,block_size=block_size,processors=number_of_processors)
        end = time.time()
        print(f"Time taken: {end-start} seconds")
        print("Error,", np.linalg.norm(x - x_sol))
    
    if cprofiler:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.sort_stats("time").print_stats(20)

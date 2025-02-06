import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, vstack, hstack, save_npz, load_npz, block_diag
from scipy.sparse.linalg import inv, spsolve
from concurrent.futures import ProcessPoolExecutor
import time
import cProfile
import pstats

def matrix_block_diagonal_inv(D, block_size):
    inv_blocks = [] 
    n_blocks = D.shape[0] // block_size
    for i in range(n_blocks):
        block = D[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size].toarray()
        inv_block = np.linalg.inv(block)
        inv_blocks.append(csr_matrix(inv_block))
    return block_diag(inv_blocks, format='csr')

# def factorize(M, f, block_size):
#     """ 
#     Factorizes matrix M and vector f by order of odd and even indices (as per cyclic reduction). 
#     Assuming P is a permutation matrix that reorders the sequence 1,...,n as 1,3,...,n,2,4,...,n-1, then
#     M = P^T * [A T; S B] * P, f = P^T * [vo; ve].

#     Parameters
#     ----------
#     M : csr_matrix
#         Matrix to factorize.
#     f : np.ndarray
#         Vector to factorize.
#     block_size : int
#         Size of the individual tridiagonal blocks. I.e. if a block is 4x4, block_size = 4.
    
#     Returns
#     -------
#     A : csr_matrix
#         Odd indices diagonal block matrix.
#     T : csr_matrix
#         Upper bidiagonal block matrix.
#     S : csr_matrix
#         Lower bidiagonal block matrix.
#     B : csr_matrix
#         Even indices diagonal block matrix.
#     vo : np.ndarray
#         Odd indices vector.
#     ve : np.ndarray
#         Even indices vector.
#     """
#     m, n = M.shape
#     number_of_diagonal_blocks = m // block_size
#     n_odd = number_of_diagonal_blocks // 2  
#     n_even = number_of_diagonal_blocks - n_odd  
    
#     B = lil_matrix((m, n))
#     g = lil_matrix((m,1))

#     start = time.time()
#     # Even indices diagonal 
#     for i in range(n_even):
#         j = 2 * i
#         B[block_size * n_odd + block_size * i:block_size * n_odd + block_size * (i + 1), block_size * n_odd + block_size * i:block_size * n_odd + block_size * (i + 1)] = M[j * block_size:(j + 1) * block_size, j * block_size:(j + 1) * block_size]
#         g[block_size * n_odd + block_size * i:block_size * n_odd + block_size * (i + 1)] = f[j * block_size:(j + 1) * block_size]

#     # Odd indices diagonal 
#     for i in range(n_odd):
#         j = 2 * i + 1
#         B[block_size * i:block_size * (i + 1), block_size * i:block_size * (i + 1)] = M[j * block_size:(j + 1) * block_size, j * block_size:(j + 1) * block_size]
#         g[block_size * i:block_size * (i + 1)] = f[j * block_size:(j + 1) * block_size]

#     # Off-diagonal blocks, even indices (F_i)
#     for i in range(n_odd):
#         j = 2 * i
#         B[block_size * n_odd + block_size * i:block_size * n_odd + block_size * (i + 1), block_size * i:block_size * (i + 1)] = M[j * block_size:(j + 1) * block_size, (j + 1) * block_size:(j + 2) * block_size]
    
#     # Off-diagonal blocks, odd indices (F_i)
#     for i in range(n_even - 1):
#         j = 2 * i + 1
#         B[block_size * i:block_size * (i + 1), block_size * n_odd + block_size * (i + 1):block_size * n_odd + block_size * (i + 2)] = M[j * block_size:(j + 1) * block_size, (j + 1) * block_size:(j + 2) * block_size]

#     # Off-diagonal blocks, even indices (E_i)
#     for i in range(n_even - 1):
#         j = 2 * i
#         B[block_size * n_odd + block_size * (i + 1):block_size * n_odd + block_size * (i + 2), block_size * i:block_size * (i + 1)] = M[(j + 2) * block_size:(j + 3) * block_size, (j + 1) * block_size:(j + 2) * block_size]

#     # Off-diagonal blocks, odd indices (E_i)
#     for i in range(n_odd):
#         j = 2 * i + 1
#         B[block_size * i:block_size * (i + 1), block_size * n_odd + block_size * i:block_size * n_odd + block_size * (i + 1)] = M[j * block_size:(j + 1) * block_size, (j - 1) * block_size:j * block_size]

#     B = B.tocsr()

#     A = B[0:n_odd * block_size, 0:n_odd * block_size]
#     T = B[0:n_odd * block_size, n_odd * block_size:]
#     S = B[n_odd * block_size:, 0:n_odd * block_size]
#     B = B[n_odd * block_size:, n_odd * block_size:]

#     vo = g[0:n_odd * block_size]
#     ve = g[n_odd * block_size:]
#     return A, T, S, B, vo, ve

def factorize(M, f, block_size):
    m, n = M.shape
    number_of_diagonal_blocks = m // block_size
    n_odd = number_of_diagonal_blocks // 2
    n_even = number_of_diagonal_blocks - n_odd

    # Preallocate data structures
    B_data = []
    B_rows = []
    B_cols = []
    g = lil_matrix((m,1))  # Using a dense array for g

        # Even indices diagonal 
    for i in range(n_even):
        j = 2 * i
        row_start = block_size * n_odd + block_size * i
        
        # Extract f as a 1D array using `toarray()` and flatten
        f_slice = f[j * block_size:(j + 1) * block_size].toarray().flatten()
        g[row_start:row_start + block_size] = f_slice  # This should work now

        # Adding diagonal blocks to B
        for row in range(block_size):
            for col in range(block_size):
                B_rows.append(row_start + row)
                B_cols.append(row_start + col)
                B_data.append(M[j * block_size + row, j * block_size + col])  # This may need review

    # Odd indices diagonal 
    for i in range(n_odd):
        j = 2 * i + 1
        row_start = block_size * i

        # Extract f as a 1D array using `toarray()` and flatten
        f_slice = f[j * block_size:(j + 1) * block_size].toarray().flatten()
        g[row_start:row_start + block_size] = f_slice  # This should work now

        # Adding diagonal blocks to B
        for row in range(block_size):
            for col in range(block_size):
                B_rows.append(row_start + row)
                B_cols.append(row_start + col)
                B_data.append(M[j * block_size + row, j * block_size + col])  # This may need review

    # Off-diagonal blocks, even indices (F_i)
    for i in range(n_odd):
        j = 2 * i
        row_start = block_size * n_odd + block_size * i
        for row in range(block_size):
            for col in range(block_size):
                B_rows.append(row_start + row)
                B_cols.append(block_size * i + col)
                B_data.append(M[j * block_size + row, (j + 1) * block_size + col])  # Accessing should be validated

    # Off-diagonal blocks, odd indices (F_i)
    for i in range(n_even - 1):
        j = 2 * i + 1
        for row in range(block_size):
            for col in range(block_size):
                B_rows.append(block_size * i + row)
                B_cols.append(block_size * n_odd + (i + 1) * block_size + col)
                B_data.append(M[j * block_size + row, (j + 1) * block_size + col])

    # Continue to construct remaining off-diagonal blocks similarly...

    # Create the CSR matrix at once
    print(M.shape)
    print(len(B_data))
    print(len(B_rows))
    print(len(B_cols))
    B = csr_matrix((B_data, (B_rows, B_cols)), shape=(m, n))
    
    # Extract the required matrices
    A = B[0:n_odd * block_size, 0:n_odd * block_size]
    T = B[0:n_odd * block_size, n_odd * block_size:]
    S = B[n_odd * block_size:, 0:n_odd * block_size]
    B = B[n_odd * block_size:, n_odd * block_size:]

    vo = g[0:n_odd * block_size]
    ve = g[n_odd * block_size:]
    return A, T, S, B, vo, ve

def create_permutation_matrix(m):
    """Creates a permutation matrix for odd/even reordering."""
    permutation = np.zeros((m, m), dtype=int)
    odd_indices = np.arange(0, m, 2)
    even_indices = np.arange(1, m, 2)
    
    # Fill permutation matrix
    for idx, val in enumerate(odd_indices):
        permutation[idx, val] = 1
    for idx, val in enumerate(even_indices):
        permutation[len(odd_indices) + idx, val] = 1
    
    return csr_matrix(permutation)

# Use this permutation in the factorization
# P = create_permutation_matrix(m)
# M_permuted = P.T @ M @ P  # Efficiently permute M


def cyclic_reduction(M, f, block_size):
    # NB! Method overwrites M, so make sure to pass a copy if the original matrix is needed later.
    if not isinstance(M, csr_matrix):
        M = csr_matrix(M)

    m, n = M.shape
    number_of_diagonal_blocks = m // block_size
    n_odd = number_of_diagonal_blocks // 2  
    n_even = number_of_diagonal_blocks - n_odd

    k = int(np.log2(number_of_diagonal_blocks)) + 1
    if k % 2 != 0:
        k -= 1

    y_j = [f]
    y_jodd = []
    x_j = []

    T_j = []
    A_jinv = []

    # Forward pass
    for j in range(k):
        A, T, S, B, vo, ve = factorize(M, y_j[j], block_size)
        A_jinv.append(matrix_block_diagonal_inv(A, block_size))
        T_j.append(T)
        V = S @ A_jinv[j]
        M = B - V @ T
        y_jodd.append(vo)
        y_j.append(ve - V @ y_jodd[j])
    
    # Solve reduced system
    x_j.append(spsolve(M, y_j[-1]))

    # Backward pass
    for j in range(k-1, -1, -1):
        x_last = x_j[-1].reshape(-1, 1)
        x_to_add = np.concatenate((A_jinv[j] @ (y_jodd[j] - T_j[j] @ x_last), x_last))

        array_block_size = len(x_to_add) // block_size
        half_size = array_block_size // 2

        first_half = x_to_add[:half_size*block_size].flatten().reshape(half_size, block_size)
        second_half = x_to_add[half_size*block_size:].flatten().reshape(array_block_size-half_size, block_size)

        woven_array = np.array([])

        for i in range(half_size):
            woven_array = np.append(woven_array,second_half[i])
            woven_array = np.append(woven_array,first_half[i])
        if len(second_half) > half_size:
            woven_array = np.append(woven_array,second_half[-1])
            
        x_to_add = woven_array
        x_j.append(x_to_add)

    return x_j[-1]


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    # Load harmonic oscillator tests. 
    # Options for N: 2, 3, 10, 16, 1k, 4k, 16k, 100k, 500k
    N = 4000
    # A = np.eye(4*N)
    # f = np.array([i for i in range(1, N + 1) for _ in range(4)])
    # x = np.linalg.solve(A, f)
    A, f, x = load_npz(f"sparse_harmonic/A_{N}.npz"), load_npz(f"sparse_harmonic/f_{N}.npz"), load_npz(f"sparse_harmonic/x_{N}.npz")
    block_size = 4

    print("Sizes A, f, u: ", A.shape, f.shape, x.shape,"\n")
    np.set_printoptions(precision=4, suppress=True)

    start = time.time()
    sol0 = cyclic_reduction(A,f,block_size)
    end = time.time()
    print(f"Error: ", np.linalg.norm(x-sol0))
    print(f"Elapsed time: {end-start} \n")

    #print("Solution: ", sol0)   

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('cumulative')  # Sort by cumulative time
    # stats.print_stats()



    

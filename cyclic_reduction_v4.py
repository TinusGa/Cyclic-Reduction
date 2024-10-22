import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, block_diag
from scipy.sparse.linalg import inv, spsolve
from concurrent.futures import ProcessPoolExecutor
import time

# Updated block diagonal inversion function to handle arbitrary block sizes
def matrix_block_diagonal_inv(D, n_blocks, block_size):
    # Initialize a list to hold the inverse blocks in sparse format (CSR)
    inv_blocks = []
    
    for i in range(n_blocks):
        # Extract the block from the CSR matrix D
        block = D[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size].toarray()  # Convert to dense for inversion
        
        # Compute the inverse of the block
        inv_block = np.linalg.inv(block)
        
        # Convert the inverse block back to CSR format and append to the list
        inv_blocks.append(csr_matrix(inv_block))
    
    # Use scipy's block_diag to create a block diagonal sparse matrix from the inverse blocks
    return block_diag(inv_blocks, format='csr')

def cyclic_reduction_parallel(A, f, block_size, max_depth=3, depth=0):
    m, n = A.shape
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)

    # Number of diagonal blocks for the given block size
    number_of_diagonal_blocks = m // block_size
    n_odd = number_of_diagonal_blocks // 2  # number of diagonal blocks with odd indices
    n_even = number_of_diagonal_blocks - n_odd  # number of diagonal blocks with even indices
    
    if depth >= max_depth:
        return spsolve(A, f)
    
    B = lil_matrix((m, n))
    g = np.zeros(m)

    # Even indices diagonal (process the even diagonal blocks first)
    for i in range(n_even):
        j = 2 * i
        B[block_size * n_odd + block_size * i:block_size * n_odd + block_size * (i + 1), block_size * n_odd + block_size * i:block_size * n_odd + block_size * (i + 1)] = A[j * block_size:(j + 1) * block_size, j * block_size:(j + 1) * block_size]
        g[block_size * n_odd + block_size * i:block_size * n_odd + block_size * (i + 1)] = f[j * block_size:(j + 1) * block_size]

    # Odd indices diagonal (process the odd diagonal blocks)
    for i in range(n_odd):
        j = 2 * i + 1
        B[block_size * i:block_size * (i + 1), block_size * i:block_size * (i + 1)] = A[j * block_size:(j + 1) * block_size, j * block_size:(j + 1) * block_size]
        g[block_size * i:block_size * (i + 1)] = f[j * block_size:(j + 1) * block_size]

    # Off-diagonal blocks, even indices (F_i)
    for i in range(n_odd):
        j = 2 * i
        B[block_size * n_odd + block_size * i:block_size * n_odd + block_size * (i + 1), block_size * i:block_size * (i + 1)] = A[j * block_size:(j + 1) * block_size, (j + 1) * block_size:(j + 2) * block_size]
    
    # Off-diagonal blocks, odd indices (F_i)
    for i in range(n_even - 1):
        j = 2 * i + 1
        B[block_size * i:block_size * (i + 1), block_size * n_odd + block_size * (i + 1):block_size * n_odd + block_size * (i + 2)] = A[j * block_size:(j + 1) * block_size, (j + 1) * block_size:(j + 2) * block_size]

    # Off-diagonal blocks, even indices (E_i)
    for i in range(n_even - 1):
        j = 2 * i
        B[block_size * n_odd + block_size * (i + 1):block_size * n_odd + block_size * (i + 2), block_size * i:block_size * (i + 1)] = A[(j + 2) * block_size:(j + 3) * block_size, (j + 1) * block_size:(j + 2) * block_size]

    # Off-diagonal blocks, odd indices (E_i)
    for i in range(n_odd):
        j = 2 * i + 1
        B[block_size * i:block_size * (i + 1), block_size * n_odd + block_size * i:block_size * n_odd + block_size * (i + 1)] = A[j * block_size:(j + 1) * block_size, (j - 1) * block_size:j * block_size]

    B = B.tocsr()

    # Extract D1, D2, F, G based on block partitions
    D1 = B[0:n_odd * block_size, 0:n_odd * block_size]
    F = B[0:n_odd * block_size, n_odd * block_size:]
    G = B[n_odd * block_size:, 0:n_odd * block_size]
    D2 = B[n_odd * block_size:, n_odd * block_size:]

    # Split the right-hand side vector `g`
    vo = g[0:n_odd * block_size]
    ve = g[n_odd * block_size:]

    # Compute inverse block-diagonal operations
    G_inv_D1 = G @ matrix_block_diagonal_inv(D1, n_odd, block_size)
    F_inv_D2 = F @ matrix_block_diagonal_inv(D2, n_even, block_size)

    if depth <= max_depth:   
        with ProcessPoolExecutor(max_workers=8) as executor:
            future_odd_res = executor.submit(cyclic_reduction_parallel, D1 - F_inv_D2 @ G, vo - F_inv_D2 @ ve, block_size, max_depth, depth + 1)  # ODDS
            future_even_res = executor.submit(cyclic_reduction_parallel, D2 - G_inv_D1 @ F, ve - G_inv_D1 @ vo, block_size, max_depth, depth + 1)  # EVENS
            odd_res = future_odd_res.result()
            even_res = future_even_res.result()

    sol_arr = []
    j = 0
    while len(sol_arr) < (len(even_res) + len(odd_res)):
        sol_arr.extend(even_res[j*block_size:(j+1)*block_size])
        sol_arr.extend(odd_res[j*block_size:(j+1)*block_size])
        j += 1

    return np.array(sol_arr)



if __name__ == "__main__":
    # Load harmonic oscillator tests
    #A, f, u = np.load("harmonic_oscillator/A_1000.npy"), np.load("harmonic_oscillator/f_1000.npy"), np.load("harmonic_oscillator/x_1000.npy")
    A, f, u = np.load("harmonic_oscillator/A_4000.npy"), np.load("harmonic_oscillator/f_4000.npy"), np.load("harmonic_oscillator/x_4000.npy")
    #A, f, u = np.load("harmonic_oscillator/A_8000.npy"), np.load("harmonic_oscillator/f_8000.npy"), np.load("harmonic_oscillator/x_8000.npy")

    A_csr = csr_matrix(A)
    block_size = 4
    print("Sizes A, f, u: ", A.shape, f.shape, u.shape,"\n")
    
    start0 = time.time()
    print("Solving with cyclic reduction, parallelism depth 0 (1 worker)")
    sol0 = cyclic_reduction_parallel(A_csr, f, block_size, max_depth=0)
    end0 = time.time()
    print(f"Error: ", np.linalg.norm(u-sol0))
    print(f"Elapsed time for cyclic reduction, parallelism: {end0-start0} \n")

    
    start1 = time.time()
    print("Solving with cyclic reduction, parallelism depth 1 (2 workers)")
    sol1 = cyclic_reduction_parallel(A_csr, f, block_size, max_depth=1)
    end1 = time.time()
    print(f"Error: ", np.linalg.norm(u-sol1))
    print(f"Elapsed time for cyclic reduction, parallelism: {end1-start1} \n")


    # start2 = time.time()
    # print("Solving with cyclic reduction, parallelism depth 2 (4 workers)")
    # sol2 = cyclic_reduction_parallel(A_csr, f, max_depth=2)
    # end2 = time.time()
    # print(f"Error: ", np.linalg.norm(u-sol2))
    # print(f"Elapsed time for cyclic reduction, parallelism: {end2-start2} \n")

    start3 = time.time()
    print("Solving with cyclic reduction, parallelism depth 3 (8 workers)")
    sol3 = cyclic_reduction_parallel(A_csr, f, block_size, max_depth=3)
    end3 = time.time()
    print(f"Error: ", np.linalg.norm(u-sol3))
    print(f"Elapsed time for cyclic reduction, parallelism: {end3-start3} \n")
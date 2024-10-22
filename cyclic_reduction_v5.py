import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, block_diag
from scipy.sparse.linalg import inv, spsolve
from concurrent.futures import ProcessPoolExecutor
import time


def matrix_block_diagonal_inv(D, n_blocks, block_size):
    inv_blocks = [] 
    for i in range(n_blocks):
        block = D[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size].toarray()
        inv_block = np.linalg.inv(block)
        inv_blocks.append(csr_matrix(inv_block))
    return block_diag(inv_blocks, format='csr')

def cyclic_reduction_parallel(A, f, block_size):
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)
    m, n = A.shape
    number_of_diagonal_blocks = m // block_size
    n_odd = number_of_diagonal_blocks // 2  
    n_even = number_of_diagonal_blocks - n_odd  
    
    B = lil_matrix((m, n))
    g = np.zeros(m)

    # Even indices diagonal 
    for i in range(n_even):
        j = 2 * i
        B[block_size * n_odd + block_size * i:block_size * n_odd + block_size * (i + 1), block_size * n_odd + block_size * i:block_size * n_odd + block_size * (i + 1)] = A[j * block_size:(j + 1) * block_size, j * block_size:(j + 1) * block_size]
        g[block_size * n_odd + block_size * i:block_size * n_odd + block_size * (i + 1)] = f[j * block_size:(j + 1) * block_size]

    # Odd indices diagonal 
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

    D1 = B[0:n_odd * block_size, 0:n_odd * block_size]
    F = B[0:n_odd * block_size, n_odd * block_size:]
    G = B[n_odd * block_size:, 0:n_odd * block_size]
    D2 = B[n_odd * block_size:, n_odd * block_size:]

    vo = g[0:n_odd * block_size]
    ve = g[n_odd * block_size:]

    G_inv_D1 = G @ matrix_block_diagonal_inv(D1, n_odd, block_size)
    F_inv_D2 = F @ matrix_block_diagonal_inv(D2, n_even, block_size)

    odd_half = [D1 - F_inv_D2 @ G, vo - F_inv_D2 @ ve]
    even_half = [D2 - G_inv_D1 @ F, ve - G_inv_D1 @ vo]

    return odd_half, even_half

def solve_cyclic_reduction(A,f,block_size,depth=0):
    if depth <= 3:
        odd_half, even_half = cyclic_reduction_parallel(A,f,block_size)
        solve_cyclic_reduction(odd_half[0],odd_half[1],block_size,depth+1)
        solve_cyclic_reduction(even_half[0],even_half[1],block_size,depth+1)
    return "Finished"



if __name__ == "__main__":
    # Load harmonic oscillator tests
    A, f, u = np.load("harmonic_oscillator/A_1000.npy"), np.load("harmonic_oscillator/f_1000.npy"), np.load("harmonic_oscillator/x_1000.npy")
    #A, f, u = np.load("harmonic_oscillator/A_4000.npy"), np.load("harmonic_oscillator/f_4000.npy"), np.load("harmonic_oscillator/x_4000.npy")
    #A, f, u = np.load("harmonic_oscillator/A_8000.npy"), np.load("harmonic_oscillator/f_8000.npy"), np.load("harmonic_oscillator/x_8000.npy")

    A_csr = csr_matrix(A)
    block_size = 4
    print("Sizes A, f, u: ", A.shape, f.shape, u.shape,"\n")

    start = time.time()
    solve_cyclic_reduction(A_csr,f,block_size)
    end = time.time()
    print(f"Elapsed time: {end-start} \n")
    
    # start0 = time.time()
    # print("Solving with cyclic reduction, parallelism depth 0 (1 worker)")
    # sol0 = cyclic_reduction_parallel(A_csr, f, block_size, max_depth=0)
    # end0 = time.time()
    # print(f"Error: ", np.linalg.norm(u-sol0))
    # print(f"Elapsed time for cyclic reduction, parallelism: {end0-start0} \n")

    
    # start1 = time.time()
    # print("Solving with cyclic reduction, parallelism depth 1 (2 workers)")
    # sol1 = cyclic_reduction_parallel(A_csr, f, block_size, max_depth=1)
    # end1 = time.time()
    # print(f"Error: ", np.linalg.norm(u-sol1))
    # print(f"Elapsed time for cyclic reduction, parallelism: {end1-start1} \n")


    # start2 = time.time()
    # print("Solving with cyclic reduction, parallelism depth 2 (4 workers)")
    # sol2 = cyclic_reduction_parallel(A_csr, f, max_depth=2)
    # end2 = time.time()
    # print(f"Error: ", np.linalg.norm(u-sol2))
    # print(f"Elapsed time for cyclic reduction, parallelism: {end2-start2} \n")

    # start3 = time.time()
    # print("Solving with cyclic reduction, parallelism depth 3 (8 workers)")
    # sol3 = cyclic_reduction_parallel(A_csr, f, block_size, max_depth=3)
    # end3 = time.time()
    # print(f"Error: ", np.linalg.norm(u-sol3))
    # print(f"Elapsed time for cyclic reduction, parallelism: {end3-start3} \n")
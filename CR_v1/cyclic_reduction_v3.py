import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, block_diag
from scipy.sparse.linalg import inv, spsolve
from concurrent.futures import ProcessPoolExecutor
import time

# def matrix_block_diagonal_inv(D, n_blocks):
#     # Initialize a list to hold the inverse blocks in sparse format (CSR)
#     inv_blocks = []
    
#     for i in range(n_blocks):
#         # Extract the 2x2 block from the CSR matrix D (keeping it sparse)
#         block = D[i*2:(i+1)*2, i*2:(i+1)*2].toarray()  # still need to convert to dense for inversion
        
#         # Compute the inverse of the block
#         inv_block = np.linalg.inv(block)
        
#         # Convert the inverse block back to CSR format and append to the list
#         inv_blocks.append(csr_matrix(inv_block))
    
#     # Use scipy's block_diag to create a block diagonal sparse matrix from the inverse blocks
#     return block_diag(inv_blocks, format='csr')

def matrix_block_diagonal_inv(D, n_blocks):
    # Initialize a list to hold the inverse blocks in sparse format (CSR)
    inv_blocks = []
    
    for i in range(n_blocks):
        # Extract the 2x2 block from the CSR matrix D (keeping it sparse)
        block = D[i*2:(i+1)*2, i*2:(i+1)*2].toarray()  # Convert the block to a dense matrix for simplicity

        # Extract the elements of the 2x2 block
        a, b = block[0, 0], block[0, 1]
        c, d = block[1, 0], block[1, 1]

        # Compute the determinant
        det = a * d - b * c

        if det == 0:
            raise np.linalg.LinAlgError("Singular matrix block found, can't invert.")

        # Compute the inverse using the adjugate and determinant
        inv_block = (1.0 / det) * np.array([[d, -b], [-c, a]])

        # Convert the inverse block back to CSR format and append to the list
        inv_blocks.append(csr_matrix(inv_block))
    
    # Use scipy's block_diag to create a block diagonal sparse matrix from the inverse blocks
    return block_diag(inv_blocks, format='csr')


def cyclic_reduction_parallel(A, f, max_depth=3, depth=0):
    m, n = A.shape
    # Check if A is already in CSR format
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)

    number_of_diagonal_blocks = m // 2  # total number of 2x2 blocks on the diagonal
    n_odd = number_of_diagonal_blocks // 2  # number of diagonal blocks with odd indices
    n_even = number_of_diagonal_blocks - n_odd  # number of diagonal blocks with even indices
            
    if depth >= max_depth:
        #return np.linalg.solve(A.toarray(), f)
        return spsolve(A, f)
    
    B = lil_matrix((m, n))
    g = np.zeros(m)

    # Even indices diagonal
    for i in range(0,n_even):
        j = 2*i
        B[2*n_odd+2*i:2*n_odd+2*i+2,2*n_odd+2*i:2*n_odd+2*i+2] = A[j*2:2*j+2,j*2:2*j+2]
        g[2*n_odd+2*i:2*n_odd+2*i+2] = f[j*2:2*j+2]

    # Odd indices diagonal
    for i in range(0,n_odd):
        j = 2*i+1
        B[2*i:2*i+2,2*i:2*i+2] = A[j*2:2*j+2,j*2:2*j+2]
        g[2*i:2*i+2] = f[j*2:2*j+2]

    # Off-diagonal blocks, even indices (F_i)
    for i in range(0,n_odd):
        j = 2*i
        B[2*n_odd+2*i:2*n_odd+2*i+2,2*i:2*i+2] = A[j*2:2*j+2,j*2+2:2*j+4]
    
    # Off-diagonal blocks, odd indices (F_i)
    for i in range(0,n_even-1):
        j = 2*i+1
        B[2*i:2*i+2,2*n_odd+2+2*i:2*n_odd+4+2*i] = A[j*2:2*j+2,j*2+2:2*j+4]

    # Off-diagonal blocks, even indices (E_i)
    for i in range(0,n_even-1):
        j = 2*i
        B[2*n_odd+2+j:2*n_odd+j+4,j:j+2] = A[j*2+4:2*j+6,j*2+2:2*j+4]

    # Off-diagonal blocks, odd indices (E_i)
    for i in range(0,n_odd):
        j = 2*i+1
        B[2*i:2*i+2,2*n_odd+2*i:2*n_odd+2+2*i] = A[j*2:2*j+2,j*2-2:2*j]

    B = B.tocsr()

    D1 = B[0:n_odd * 2, 0:n_odd * 2]
    F = B[0:n_odd * 2, n_odd * 2:]
    G = B[n_odd * 2:, 0:n_odd * 2]
    D2 = B[n_odd * 2:, n_odd * 2:]

    vo = g[0:n_odd * 2]
    ve = g[n_odd * 2:]

    G_inv_D1 = G @ matrix_block_diagonal_inv(D1, n_odd)
    F_inv_D2 = F @ matrix_block_diagonal_inv(D2, n_even)

    if depth <= max_depth:   
        with ProcessPoolExecutor(max_workers=8) as executor:
            future_odd_res = executor.submit(cyclic_reduction_parallel, D1 - F_inv_D2 @ G, vo - F_inv_D2 @ ve, max_depth, depth+1) # ODDS
            future_even_res = executor.submit(cyclic_reduction_parallel, D2 - G_inv_D1 @ F, ve - G_inv_D1 @ vo, max_depth, depth+1) # EVENS
            odd_res = future_odd_res.result()
            even_res = future_even_res.result()
    
    sol_arr = np.zeros_like(f)
    
    sol_arr[0::4] = even_res[0::2]
    sol_arr[1::4] = even_res[1::2]
    sol_arr[2::4] = odd_res[0::2]
    sol_arr[3::4] = odd_res[1::2]

    return sol_arr

if __name__ == "__main__":
    A,f,u = np.load("test_cases/Matrix_E.npy"), np.load("test_cases/vector_j.npy"), np.load("test_cases/solution_y.npy")
    #A,f,u = np.load("test_cases/Matrix_D.npy"), np.load("test_cases/vector_i.npy"), np.load("test_cases/solution_x.npy")
    A_csr = csr_matrix(A)
    print("Sizes A, f, u: ", A.shape, f.shape, u.shape,"\n")

    start = time.time()
    print("Solving with numpy")
    #sol = np.linalg.solve(A, f)
    sol = spsolve(A_csr, f)
    end = time.time()
    print(f"Error: ", np.linalg.norm(u-sol))
    print(f"Elapsed time for numpy solve: {end-start} \n")

    start0 = time.time()
    print("Solving with cyclic reduction, parallelism depth 0 (1 worker)")
    sol0 = cyclic_reduction_parallel(A_csr, f, max_depth=0)
    end0 = time.time()
    print(f"Error: ", np.linalg.norm(u-sol0))
    print(f"Elapsed time for cyclic reduction, parallelism: {end0-start0} \n")

    
    # start1 = time.time()
    # print("Solving with cyclic reduction, parallelism depth 1 (2 workers)")
    # sol1 = cyclic_reduction_parallel(A_csr, f, max_depth=1)
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
    # sol3 = cyclic_reduction_parallel(A_csr, f, max_depth=3)
    # end3 = time.time()
    # print(f"Error: ", np.linalg.norm(u-sol3))
    # print(f"Elapsed time for cyclic reduction, parallelism: {end3-start3} \n")
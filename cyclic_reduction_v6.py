import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, vstack, hstack, save_npz, load_npz, block_diag
from scipy.sparse.linalg import inv, spsolve
from concurrent.futures import ProcessPoolExecutor
import time
import cProfile
import pstats

# HELPER FUNCTIONS
def matrix_block_diagonal_inv(D, block_size):
    inv_blocks = [] 
    n_blocks = D.shape[0] // block_size
    for i in range(n_blocks):
        block = D[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size].toarray()
        inv_block = np.linalg.inv(block)
        inv_blocks.append(csr_matrix(inv_block))
    return block_diag(inv_blocks, format='csr')

def create_full_permutation_matrix(m, block_size):
    # Total number of blocks
    num_blocks = m // block_size  # For m = 12, block_size = 4, num_blocks = 3

    # Prepare lists to hold the data for the sparse matrix
    data = []
    rows = []
    cols = []

    # Define the permutation order: [Block 1, Block 0, Block 2]
    perm_order = []
    for i in range(num_blocks):
        if i % 2 == 1:  # Append odd blocks first
            perm_order.append(i)
    for i in range(num_blocks):
        if i % 2 == 0:  # Append even blocks next
            perm_order.append(i)

    # Fill the sparse matrix data
    for new_index, old_index in enumerate(perm_order):
        start_new = new_index * block_size
        start_old = old_index * block_size
        for i in range(block_size):
            # Set the value for the non-zero entries
            data.append(1)  # Value to insert (identity matrix entries)
            rows.append(start_new + i)  # Row index in new position
            cols.append(start_old + i)   # Column index in old position

    # Create the sparse matrix using the collected data
    Q = csr_matrix((data, (rows, cols)), shape=(m, m))
    return Q

def construct_sparse_block_tridiagonal(current_a, current_b, current_c, block_size):
    p = len(current_a)  # Number of blocks along the diagonal
    n = p * block_size  # Dimension of the final matrix
    
    data = []
    rows = []
    cols = []
    # Main diagonal blocks
    for i in range(p):
        row_offset = i * block_size
        col_offset = i * block_size
        block = current_a[i]
        for r in range(block_size):
            for c in range(block_size):
                if block[r, c] != 0:
                    data.append(block[r, c])
                    rows.append(row_offset + r)
                    cols.append(col_offset + c)

    # Upper diagonal blocks
    for i in range(p - 1):
        row_offset = i * block_size
        col_offset = (i + 1) * block_size
        block = current_b[i]
        
        for r in range(block_size):
            for c in range(block_size):
                if block[r, c] != 0:
                    data.append(block[r, c])
                    rows.append(row_offset + r)
                    cols.append(col_offset + c)

    # Lower diagonal blocks
    for i in range(1, p):
        row_offset = i * block_size
        col_offset = (i - 1) * block_size
        block = current_c[i - 1]
        
        for r in range(block_size):
            for c in range(block_size):
                if block[r, c] != 0:
                    data.append(block[r, c])
                    rows.append(row_offset + r)
                    cols.append(col_offset + c)

    # Construct the sparse matrix in CSR format
    block_tridiagonal_matrix = csr_matrix((data, (rows, cols)), shape=(n, n))
    return block_tridiagonal_matrix

def extract_block_diagonals(matrix, block_size):
    """
    Extracts the main, upper, and lower block diagonals from a block tridiagonal matrix.
    
    Parameters:
    - matrix: A sparse matrix in CSR or CSC format representing a block tridiagonal matrix
    - block_size: The size of each block (assuming square blocks)
    
    Returns:
    - main_diagonal_blocks: List of blocks from the main diagonal
    - upper_diagonal_blocks: List of blocks from the upper diagonal
    - lower_diagonal_blocks: List of blocks from the lower diagonal
    """
    n = matrix.shape[0]  # Total size of the matrix
    p = n // block_size  # Number of blocks along each diagonal

    # Initialize lists for each diagonal
    main_diagonal_blocks = []
    upper_diagonal_blocks = []
    lower_diagonal_blocks = []

    # Extract main diagonal blocks
    for i in range(p):
        row_start = i * block_size
        row_end = row_start + block_size
        col_start = i * block_size
        col_end = col_start + block_size
        main_block = matrix[row_start:row_end, col_start:col_end].toarray()
        main_diagonal_blocks.append(main_block)

    # Extract upper diagonal blocks
    for i in range(p - 1):
        row_start = i * block_size
        row_end = row_start + block_size
        col_start = (i + 1) * block_size
        col_end = col_start + block_size
        upper_block = matrix[row_start:row_end, col_start:col_end].toarray()
        upper_diagonal_blocks.append(upper_block)

    # Extract lower diagonal blocks
    for i in range(1, p):
        row_start = i * block_size
        row_end = row_start + block_size
        col_start = (i - 1) * block_size
        col_end = col_start + block_size
        lower_block = matrix[row_start:row_end, col_start:col_end].toarray()
        lower_diagonal_blocks.append(lower_block)

    return main_diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks

def split_vector_into_blocks(vector, block_size):
    """
    Splits a vector into blocks of a specified size.
    
    Parameters:
    - vector: The input vector (1D array or list)
    - block_size: The size of each block
    
    Returns:
    - A list where each element is an array of size `block_size`
    """
    # Ensure vector is a numpy array
    #vector = np.array(vector).flatten()
    #print(vector.shape)
    
    # Number of blocks
    num_blocks = vector.shape[0] // block_size
    
    # Extract blocks
    blocks = [vector[i * block_size:(i + 1) * block_size] for i in range(num_blocks)]
    
    return blocks

def create_Zp_matrix(p):
    # Create an empty matrix of zeros
    #Z_p = np.zeros(((p-1)*block_size, (2*p-2)*block_size))
    z_p = np.hstack((np.eye((block_size)), np.eye((block_size))))
    Z_p = block_diag([z_p for _ in range(p-1)]) 
    
    return Z_p

def factorize(M, f, block_size):
    m, n = M.shape
    number_of_diagonal_blocks = m // block_size
    n_odd = number_of_diagonal_blocks // 2
    n_even = number_of_diagonal_blocks - n_odd

    Q = create_full_permutation_matrix(m, block_size)
    G = Q @ M @ Q.T
    g = Q @ f
    
    # Extract the required matrices
    A = G[0:n_odd * block_size, 0:n_odd * block_size]
    T = G[0:n_odd * block_size, n_odd * block_size:]
    S = G[n_odd * block_size:, 0:n_odd * block_size]
    B = G[n_odd * block_size:, n_odd * block_size:]
    

    vo = g[0:n_odd * block_size]
    ve = g[n_odd * block_size:]
    return A, T, S, B, vo, ve

# CYCLIC REDUCTION STEPS
def cyclic_reduction_forward_step(M, f, k, block_size):
    # NB! Method overwrites M, so make sure to pass a copy if the original matrix is needed later.
    if not isinstance(M, csr_matrix):
        M = csr_matrix(M)

    m, n = M.shape
    number_of_diagonal_blocks = m // block_size
    n_odd = number_of_diagonal_blocks // 2  
    n_even = number_of_diagonal_blocks - n_odd

    y_j = [f]
    y_jodd = []

    T_j = []
    A_jinv = []

    # Forward pass

    for i in range(k):
        A, T, S, B, vo, ve = factorize(M, y_j[i], block_size)
        A_jinv.append(matrix_block_diagonal_inv(A, block_size))
        T_j.append(T)
        V = S @ A_jinv[i]
        M = B - V @ T   
        y_jodd.append(vo)
        y_j.append(ve - V @ y_jodd[i])

    return y_j, y_jodd, T_j, A_jinv, M

def cyclic_reduction_intermediate_step(y_j, M, p):
    Z_p = create_Zp_matrix(p)
    M_k = Z_p @ M @ Z_p.T
    y_k = Z_p @ y_j
    return spsolve(M_k, y_k)

def cyclic_reduction_backward_step(x_j, y_jodd, T_j, A_jinv, k,  block_size):
    x_last = x_j
    # Backward pass
    for j in range(k-1, -1, -1):
        x_to_add = np.concatenate((A_jinv[j] @ (y_jodd[j] - T_j[j] @ x_last), x_last))
        q = int(A_jinv[j].shape[0] + x_last.shape[0])
        Q = create_full_permutation_matrix(q, block_size)
        x_last = Q.T @ x_to_add
    return x_last

# CYCLIC REDUCTION PARALLEL
def cyclic_redcution_parallel(M, f, p, block_size):
    # p = number of processes
    if not isinstance(M, csr_matrix):
        M = csr_matrix(M)

    m,q = M.shape
    assert m == q, "Matrix must be square"

    n = m//block_size
    h = (n+1)//p
    print("n: ", n, "h: ", h, "\n")
    
    # 3.7
    y0_p = [] # y0 for each process (list of vectors); y0_p[i] = y0 for process i
    f = f.T.toarray().flatten()
    for i in range(1,p+1): 
        if i == 1: 
            y0 = np.array([])
            for j in range(h): 
                y0 = np.concatenate((y0, f[((i-1)*h+j)*block_size:((i-1)*h+j+1)*block_size]))
        elif i == p:
            y0 = np.zeros(block_size)
            for j in range(h): 
                y0 = np.concatenate((y0, f[((i-1)*h+j)*block_size:((i-1)*h+j+1)*block_size]))
        else:
            y0 = np.zeros(block_size)
            for j in range(h): 
                y0 = np.concatenate((y0, f[((i-1)*h+j)*block_size:((i-1)*h+j+1)*block_size]))
        y0_p.append(y0)

    # 3.3
    M_p = [] # M for each process (list of matrices); M_p[i] = M for process i
    a, c, b = extract_block_diagonals(M, block_size) 
    for i in range(1,p+1):      
        current_a = [] # Main diagonal blocks
        current_c = [] # Upper diagonal blocks
        current_b = [] # Lower diagonal blocks
        if i == 1: 
            for j in range(h-1): 
                current_a.append(a[(i-1)*h+j])
                current_b.append(b[(i-1)*h+j])
                current_c.append(c[(i-1)*h+j])
            current_a.append(a[i*h-1])
            M_p.append(construct_sparse_block_tridiagonal(current_a, current_c, current_b, block_size))
        elif i == p:
            current_a.append(np.zeros((block_size,block_size)))
            for j in range(h-1): 
                current_a.append(a[(i-1)*h+j])
                current_b.append(b[(i-1)*h+j-1])
                current_c.append(c[(i-1)*h+j-1])
            M_p.append(construct_sparse_block_tridiagonal(current_a, current_c, current_b, block_size))
        else:
            current_a.append(np.zeros((block_size,block_size)))
            for j in range(h): 
                current_a.append(a[(i-1)*h+j])
                current_b.append(b[(i-1)*h+j-1])
                current_c.append(c[(i-1)*h+j-1])
            M_p.append(construct_sparse_block_tridiagonal(current_a, current_c, current_b, block_size))

    # FORWARD PASS
    M_k_p = []
    y_j_p = np.array([])
    y_jodd_p = []
    T_j_p = []
    A_jinv_p = []
    k = int(np.log2(h))
    for i in range(p):
        y_j, y_jodd, T_j, A_jinv, M = cyclic_reduction_forward_step(M_p[i], y0_p[i], k , block_size)
        y_jodd_p.append(y_jodd)
        T_j_p.append(T_j)
        A_jinv_p.append(A_jinv)
        M_k_p.append(M)
        y_j_p = np.concatenate((y_j_p, y_j[-1]))
    
    # 3.6
    M_k = block_diag(M_k_p)

    # 3.10
    Z_p = create_Zp_matrix(p)
    M_k = Z_p @ M_k @ Z_p.T
    y_k = Z_p @ y_j_p
    
    # 3.11
    x_k = spsolve(M_k, y_k) 

    # 3.12
    x_k = Z_p.T @ x_k
    x_k_list = [x_k[:block_size]]
    for i in range (1,p-1):
        x_k_list.append(x_k[(i-1)*block_size:(i+1)*block_size])
    x_k_list.append(x_k[(p-2)*block_size:(p-1)*block_size])

    # BACKWARD PASS
    x_p = []
    for x_k, y_jodd, T_j, A_jinv in zip(x_k_list, y_jodd_p, T_j_p, A_jinv_p):
        sol_x = cyclic_reduction_backward_step(x_k, y_jodd, T_j, A_jinv, k, block_size)
        x_p.append(sol_x)
    
    print("--------------------")
    for el in x_p:
        print(el.shape)
        print(el.flatten(),"\n")   
    print("--------------------")
    
    # 3.13
    x = lil_matrix((m,1))
    for i in range(1,p+1):
        for j in range(h-1):
            x[((i-1)*h+j)*block_size:((i-1)*h+j+1)*block_size] = x_p[i-1][j*block_size:(j+1)*block_size]
    for i in range(1,p):
        x[i*h*block_size:(i*h+1)*block_size] = x_p[i][(h-2)*block_size:(h-1)*block_size]
        # print("x: ", i*h*block_size,(i*h+1)*block_size)
        # print("x_p: ", (h-1)*block_size,(h)*block_size)
        # print("\n")
        


    # Final solution
    # x[:h*block_size] = x_p[0]
    # for i in range(1,p):
    #     x[i*h*block_size:(i+1)*h*block_size] = x_p[i][block_size:]


    return x

    
if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    # Load harmonic oscillator tests. 
    # Options for N: 2, 3, 10, 16, 1k, 4k, 16k, 100k, 500k
    #N = 64
    # A = np.eye(4*N)
    # f = np.array([i for i in range(1, N + 1) for _ in range(4)])
    # x = np.linalg.solve(A, f)

    np.set_printoptions(precision=4, suppress=True)
    # TESTS! 2 PROCESSES
    block_size = 4
    # Ns = [15,31,63,127,255,511,1023,2047,4095,8191]
    # Ns_heavy = [16383,32767,65535,131071,262143,524287]
    # processes = [2,4,8]
    Ns = [15]
    processes = [8]

    for n in Ns:
        for p in processes:
            print(f"Number of processes p: {p}, N: {n}")
            A, f, x = load_npz(f"sparse_harmonic/A_{n}.npz"), load_npz(f"sparse_harmonic/f_{n}.npz"), load_npz(f"sparse_harmonic/x_{n}.npz")
            #A, f, x = load_npz(f"sparse_harmonic/A_test.npz"), load_npz(f"sparse_harmonic/f_test.npz"), load_npz(f"sparse_harmonic/x_test.npz")
            print("Sizes A, f, u: ", A.shape, f.shape, x.shape)
            sol = cyclic_redcution_parallel(A,f,p,block_size)
            print(f"Error: ", np.linalg.norm(x.toarray()-sol.T))

            print("Real solution: ", x.toarray().flatten(),"\n") 
            print("Computed solution: ", sol.toarray().flatten())
            print("\n")



    # start = time.time()
    # sol0 = cyclic_reduction(A,f,block_size)
    # end = time.time()
    # print(f"Error: ", np.linalg.norm(x-sol0))
    # print(f"Elapsed time: {end-start} \n")
    # profiler.disable()
    # profiler.print_stats(sort='cumtime')




    

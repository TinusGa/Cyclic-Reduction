import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, vstack, hstack, save_npz, load_npz, block_diag
from scipy.sparse.linalg import inv, spsolve
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, shared_memory
import time
import cProfile
import pstats

# HELPER FUNCTIONS
def matrix_block_diagonal_inv(D, block_size):
    inv_blocks = [] 
    n_blocks = D.shape[0] // block_size
    for i in range(n_blocks):
        block = D[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size].toarray()
        # Check if the block is singular
        if np.linalg.matrix_rank(block) < block_size:
            print(f"Block is singular.")
            inv_block = np.zeros((block_size, block_size))
        else:
            inv_block = np.linalg.inv(block)
        inv_blocks.append(csr_matrix(inv_block))
    return block_diag(inv_blocks, format='csr')

def create_full_permutation_matrix(m, block_size):
    # Total number of blocks
    num_blocks = m // block_size

    # Prepare lists to hold the data for the sparse matrix
    data = []
    rows = []
    cols = []

    # Define the permutation order: [Odd blocks first, then even blocks]
    perm_order = []
    for i in range(num_blocks):
        if i % 2 == 0:  # Append even blocks first
            perm_order.append(i)
    for i in range(num_blocks):
        if i % 2 == 1:  # Append odd blocks next
            perm_order.append(i)

    # Fill the sparse matrix data
    for new_index, old_index in enumerate(perm_order):
        start_new = new_index * block_size
        start_old = old_index * block_size
        for i in range(block_size):
            # Set the value for the non-zero entries
            data.append(1)  # Value to insert (identity matrix entries)
            rows.append(start_new + i)  # Row index in new position
            cols.append(start_old + i)  # Column index in old position

    # Create the sparse matrix using the collected data
    Q = csr_matrix((data, (rows, cols)), shape=(m, m))
    return Q

def create_full_permutation_matrix_inner(m, block_size):
    # Total number of blocks
    num_blocks = m // block_size

    # Define the permutation order: [Odd blocks first, 0 block, then even blocks]
    perm_order = []
    
    # Append odd-indexed blocks first
    for i in range(1, num_blocks, 2):
        perm_order.append(i)
    
    # Append the 0-th block next
    perm_order.append(0)
    
    # Append even-indexed blocks (other than 0) next
    for i in range(2, num_blocks, 2):
        perm_order.append(i)

    # Prepare lists for the sparse matrix construction
    data = []
    rows = []
    cols = []

    # Fill the sparse matrix data based on the permutation order
    for new_index, old_index in enumerate(perm_order):
        start_new = new_index * block_size
        start_old = old_index * block_size
        for i in range(block_size):
            # Set the value for the non-zero entries
            data.append(1)              # Value to insert (identity matrix entries)
            rows.append(start_new + i)   # Row index in new position
            cols.append(start_old + i)   # Column index in old position

    # Create the sparse matrix using the collected data
    P = csr_matrix((data, (rows, cols)), shape=(m, m))
    return P

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

def create_Zp_matrix(p,block_size):
    # Create an empty matrix of zeros
    z_p = np.hstack((np.eye((block_size)), np.eye((block_size))))
    diag_list = [z_p for _ in range(p-1)]
    #diag_list.extend([np.eye(block_size)])
    #diag_list.append([np.eye(block_size)])
    Z_p = block_diag(diag_list) 
    return Z_p

def create_Zp_matrix1(p, block_size):
    # Create identity and zero blocks
    I = np.eye(block_size)
    zero_block = np.zeros((block_size, block_size))
    
    # Initialize a list to hold each row as a list of blocks
    rows = []
    
    # Create the rows for the Z_p matrix based on the pattern you described
    for i in range(p+1):
        row = []
        for j in range(2 * p ):
            if (i == 0 and j == 0) or (i > 0 and (j == 2 * i - 1 or j == 2 * i)):
                # Add an identity matrix at the required positions
                row.append(I)
            else:
                # Otherwise, add a zero matrix
                row.append(zero_block)
        # Append the constructed row to the rows list
        rows.append(np.hstack(row))
    
    # Stack all rows vertically to form the full matrix
    Z_p = np.vstack(rows)
    
    return Z_p

def create_Zp_matrix2(p, block_size):
    # Create identity and zero blocks
    I = np.eye(block_size)
    zero_block = np.zeros((block_size, block_size))
    
    # Initialize a list to hold each row as a list of blocks
    rows = []
    
    # Create the rows for the Z_p matrix based on the pattern
    for i in range(p):
        row = []
        for j in range(2 * p - 2):
            # Add identity matrices at specific positions based on the pattern
            if (i == 0 and (j == 0 or j == 1)) or \
               (i == p - 1 and (j == 2 * (p - 1) - 2 or j == 2 * (p - 1) - 1)) or \
               (i > 0 and i < p - 1 and j == 2 * i):
                row.append(I)
            else:
                row.append(zero_block)
        # Append the constructed row to the rows list
        rows.append(np.hstack(row))
    
    # Stack all rows vertically to form the full matrix
    Z_p = np.vstack(rows)
    
    return Z_p

def factorize(M, f, block_size):
    m, n = M.shape
    number_of_diagonal_blocks = m // block_size
    n_even = number_of_diagonal_blocks // 2
    n_odd = number_of_diagonal_blocks - n_even

    Q = create_full_permutation_matrix_inner(m, block_size)

    G = Q @ M @ Q.T
    g = Q @ f
    A = G[0:n_even * block_size, 0:n_even * block_size]
    T = G[0:n_even * block_size, n_even * block_size:]
    S = G[n_even * block_size:, 0:n_even * block_size]
    B = G[n_even * block_size:, n_even * block_size:]

    vo = g[0:n_even * block_size]
    ve = g[n_even * block_size:]

        
    return Q, A, T, S, B, vo, ve

# CYCLIC REDUCTION STEPS
def cyclic_reduction_forward_step(M, f, k, block_size):
    

    y_j = [f]
    y_jodd = []

    T_j = []
    A_jinv = []

    Q_j = []
    # Forward pass

    for i in range(k):
        Q, A, T, S, B, vo, ve = factorize(M, y_j[i],  block_size)
        A_jinv.append(matrix_block_diagonal_inv(A, block_size))
        T_j.append(T)
        V = S @ A_jinv[i]
        M = B - V @ T   
        y_jodd.append(vo)
        y_j.append(ve - V @ y_jodd[i])
        Q_j.append(Q)

    return Q_j, y_j, y_jodd, T_j, A_jinv, M

def cyclic_reduction_backward_step(x_j, Q_j, y_jodd, T_j, A_jinv, k):
    x_last = x_j
    # Backward pass
    for j in range(k-1, -1, -1):
        x_to_add = np.concatenate((A_jinv[j] @ (y_jodd[j] - T_j[j] @ x_last), x_last))
        # q = int(A_jinv[j].shape[0] + x_last.shape[0])
        # Q = create_full_permutation_matrix(q, block_size)
        x_last = Q_j[j].T @ x_to_add
    return x_last

def get_index_f(i,block_size):
    return (i)*block_size,(i+1)*block_size

def get_index_main_M(i,block_size):
    return (i)*block_size,(i+1)*block_size, (i)*block_size,(i+1)*block_size

def get_index_upper_M(i,block_size):
    return (i-1)*block_size,(i)*block_size, (i)*block_size,(i+1)*block_size

def get_index_lower_M(i,block_size):
    return (i)*block_size,(i+1)*block_size, (i-1)*block_size,(i)*block_size

# CYCLIC REDUCTION PARALLEL
def cyclic_redcution_parallel(M, f, p, block_size):
    # p = number of processes
    if not isinstance(M, csr_matrix):
        M = csr_matrix(M)

    m,q = M.shape
    assert m == q, "Matrix must be square"

    n = m//block_size - 1
    r = int(np.log2(p))
    k = int(np.log2(n))
    h = int(2**(k - r))
    
    # 3.7
    y0_p = [] # y0 for each process (list of vectors); y0_p[i] = y0 for process i
    f = f.T.toarray().flatten()


    for i in range(p):
        if i == 0:
            index_1, index_2 = get_index_f(0,block_size)
            y0 = f[index_1:index_2]
            for j in range(1,h):
                index_1, index_2 = get_index_f(i*h+j,block_size)
                y0 = np.concatenate((y0, f[index_1:index_2]))
            y0 = np.concatenate((y0, np.zeros(block_size)))
        elif i == p-1:
            index_1, index_2 = get_index_f(i*h,block_size)
            y0 = f[index_1:index_2]
            for j in range(1,h+1):
                index_1, index_2 = get_index_f(i*h+j,block_size)
                y0 = np.concatenate((y0, f[index_1:index_2]))
        else:
            index_1, index_2 = get_index_f(i*h,block_size)
            y0 = f[index_1:index_2]
            for j in range(1,h):
                index_1, index_2 = get_index_f(i*h+j,block_size)
                y0 = np.concatenate((y0, f[index_1:index_2]))
            y0 = np.concatenate((y0, np.zeros(block_size)))
        y0_p.append(y0)
           
    # 3.3
    M_p = [] # M for each process (list of matrices); M_p[i] = M for process i
    for i in range(p):
        current_a = []
        current_c = []
        current_b = []
        if i == 0:
            index_1, index_2, index_3, index_4 = get_index_main_M(0,block_size)
            current_a.append(M[index_1:index_2,index_3:index_4])
        else:
            current_a.append(np.zeros((block_size,block_size)))

        for j in range(1,h+1):
            index_1, index_2, index_3, index_4 = get_index_main_M(i*h+j,block_size)
            current_a.append(M[index_1:index_2,index_3:index_4])
            index_1, index_2, index_3, index_4 = get_index_upper_M(i*h+j,block_size)
            current_c.append(M[index_1:index_2,index_3:index_4])
            index_1, index_2, index_3, index_4 = get_index_lower_M(i*h+j,block_size)
            current_b.append(M[index_1:index_2,index_3:index_4])
        M_p.append(construct_sparse_block_tridiagonal(current_a, current_c, current_b, block_size))

    # FORWARD PASS
    M_k_p = []
    y_j_p = np.array([])
    y_jodd_p = []
    T_j_p = []
    A_jinv_p = []
    Q_p = []
    
    #assert True == False, "stop here"
    #print("k-r: ", k-r, "\n")   
    for i in range(p):
        Q_j, y_j, y_jodd, T_j, A_jinv, M = cyclic_reduction_forward_step(M_p[i], y0_p[i], k-r, block_size)
        Q_p.append(Q_j)
        y_jodd_p.append(y_jodd)
        T_j_p.append(T_j)
        A_jinv_p.append(A_jinv)
        M_k_p.append(M)
        y_j_p = np.concatenate((y_j_p, y_j[-1]))
        
    M_k = block_diag(M_k_p)

    # 3.10
    #Z_p = csr_matrix(create_Zp_matrix1(p,block_size))
    Z_p = create_Zp_matrix1(p,block_size)  

    M_k = Z_p @ M_k @ Z_p.T
    y_k = Z_p @ y_j_p
    
    # 3.11
    x_k = spsolve(M_k, y_k) 

    # 3.12
    #x_k = Z_p.T @ x_k
    #print("x_k: ", x_k.flatten(), "\n")
    # index_1, index_2 = get_index_f(1,block_size)
    x_k_list = []
    for i in range (0,p):
        index_1 = get_index_f(i,block_size)[0]
        index_2 = get_index_f(i+1,block_size)[1]
        x_k_list.append(x_k[index_1:index_2])

    # BACKWARD PASS
    x_p = []
    for x_k, Q_j, y_jodd, T_j, A_jinv in zip(x_k_list,Q_p, y_jodd_p, T_j_p, A_jinv_p):
        sol_x = cyclic_reduction_backward_step(x_k, Q_j, y_jodd, T_j, A_jinv, k-r)
        x_p.append(sol_x)
    
    # 3.13
    x = np.array([])
    for i in range(p):
        if i == 0:
            x = np.concatenate((x, x_p[i]))
        else:
            x = np.concatenate((x, x_p[i][block_size:]))
    return x

def scalar_cyclic_reduction(M,f,block_size):
    if not isinstance(M, csr_matrix):
        M = csr_matrix(M)
    #f = f.T
    m,q = M.shape
    assert m == q, "Matrix must be square"

    n = m//block_size
    h = (n+1)//1

    k = int(np.log2(h))-2
    print("k scalar: ", k, "\n")
    Q_j, y_j, y_jodd, T_j, A_jinv, M = cyclic_reduction_forward_step(M, f, k , block_size)
    x_k = spsolve(M, y_j[-1])

    x_k = x_k.reshape(-1,1)

    sol_x = cyclic_reduction_backward_step(x_k, Q_j, y_jodd, T_j, A_jinv, k)

    return sol_x


    
if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    # Load harmonic oscillator tests. 
    np.set_printoptions(precision=4, suppress=True)
    # TESTS! 2 PROCESSES
    block_size = 4
    #Ns = [17,33,129,257,513,1025,2049,4097,8193]
    #Ns_heavy = [16385,32769,65537,131073,262145,524289]
    #processes = [2,4,8]
    Ns = [4097]
    processes = [8]

    for n in Ns:
        for p in processes:
            print(f"Number of processes p: {p}, N: {n}")
            A, f, x = load_npz(f"sparse_harmonic_new/A_{n}.npz"), load_npz(f"sparse_harmonic_new/f_{n}.npz"), load_npz(f"sparse_harmonic_new/x_{n}.npz")

            #A, f, x = load_npz(f"sparse_harmonic/A_test2.npz"), load_npz(f"sparse_harmonic/f_test2.npz"), load_npz(f"sparse_harmonic/x_test2.npz")
            print("Sizes A, f, u: ", A.shape, f.shape, x.shape)

            #sol1 = scalar_cyclic_reduction(A,f,block_size)
            #print(f"Error scalar: ", np.linalg.norm(x.toarray()-sol1.T))

            sol = cyclic_redcution_parallel(A,f,p,block_size)
            print(f"Error parallel: ", np.linalg.norm(x.toarray()-sol.T))

            print("eps: ", np.finfo(np.float64).eps)

            #print("Real solution: ", x.toarray().flatten(),"\n") 
            #print("Computed solution: ", sol.flatten())



    # profiler.disable()
    # profiler.print_stats(sort='cumtime')




    

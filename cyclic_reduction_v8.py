import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, vstack, hstack, save_npz, load_npz, block_diag
from scipy.sparse.linalg import inv, spsolve, splu
from scipy.linalg import lu
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, shared_memory
import matplotlib.pyplot as plt
import time
import cProfile
import pstats
from memory_profiler import profile

# HELPER FUNCTIONS

def matrix_block_diagonal_inv(D, block_size):
    inv_blocks = [] 
    n_blocks = D.shape[0] // block_size
    for i in range(n_blocks):
        block = D[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size].toarray()
        inv_block = np.linalg.inv(block)
        inv_blocks.append(inv_block)
    return block_diag(inv_blocks, format='csr')

def create_full_permutation_matrix(m, block_size):
    num_blocks = m // block_size
    perm_order = []
    
    for i in range(1, num_blocks, 2):
        perm_order.append(i)
    perm_order.append(0)
    for i in range(2, num_blocks, 2):
        perm_order.append(i)

    data, rows, cols = [], [], []

    for new_index, old_index in enumerate(perm_order):
        start_new = new_index * block_size
        start_old = old_index * block_size
        for i in range(block_size):
            data.append(1)              
            rows.append(start_new + i)  
            cols.append(start_old + i)  

    P = csr_matrix((data, (rows, cols)), shape=(m, m))
    return P


def create_Zp_matrix(p, block_size):
    I = csr_matrix(np.eye(block_size))        
    zero_block = csr_matrix((block_size, block_size)) 
    rows = []

    for i in range(p + 1):
        row = []
        for j in range(2 * p):
            if (i == 0 and j == 0) or (i > 0 and (j == 2 * i - 1 or j == 2 * i)):
                row.append(I)
            else:
                row.append(zero_block)
        rows.append(hstack(row))

    Z_p_sparse = vstack(rows).tocsr()
    return Z_p_sparse

def factorize(M, f, block_size):
    m = M.shape[0]
    number_of_diagonal_blocks = m // block_size
    n_even = number_of_diagonal_blocks // 2

    Q = create_full_permutation_matrix(m, block_size)
    G = Q @ M @ Q.T
    
    #LU = splu(G)
    g = Q @ f
    A = G[0:n_even * block_size, 0:n_even * block_size]
    T = G[0:n_even * block_size, n_even * block_size:]
    S = G[n_even * block_size:, 0:n_even * block_size]
    B = G[n_even * block_size:, n_even * block_size:]

    vo = g[0:n_even * block_size]
    ve = g[n_even * block_size:]

        
    return Q, A, T, S, B, vo, ve



# CYCLIC REDUCTION STEPS
def cyclic_reduction_forward_step(M, f, p, k, h, index, block_size):
    """ 
    Cyclic reduction forward step for a single process

    Parameters
        M: sparse matrix
        f: RHS vector
        p: number of processes
        k: number of reduction steps
        h: shape of submatrices and subvectors (h+1)x(h+1) and (h+1) respectively
        index: index of the process
    """
    # Determine submatrix splitting for processor "index"
    data, row, col = [], [], []
    if index == 0:
        # main_index = get_index_main_M(0, block_size)
        # block_a = M[main_index[0]:main_index[1], main_index[2]:main_index[3]].tocoo()
        # data.extend(block_a.data)
        # row.extend(block_a.row)
        # col.extend(block_a.col)
        row_start, row_end, col_start, col_end = get_index_main_M(0, block_size)

        for r in range(row_start, row_end):
            row_data_start = M.indptr[r]            
            row_data_end = M.indptr[r + 1]          
            row_columns = M.indices[row_data_start:row_data_end]  
            row_values = M.data[row_data_start:row_data_end]     

            for i, c in enumerate(row_columns):
                if col_start <= c < col_end:
                    data.append(row_values[i])
                    row.append(r - row_start)       
                    col.append(c - col_start)

    for j in range(1, h + 1):
        # Main diagonal block indices
        row_start, row_end, col_start, col_end = get_index_main_M(index * h + j, block_size)
        for r in range(row_start, row_end):
            row_data_start = M.indptr[r]
            row_data_end = M.indptr[r + 1]
            row_columns = M.indices[row_data_start:row_data_end]
            row_values = M.data[row_data_start:row_data_end]
            for i, c in enumerate(row_columns):
                if col_start <= c < col_end:
                    data.append(row_values[i])
                    row.append((r - row_start) + j * block_size)
                    col.append((c - col_start) + j * block_size)

        # Upper diagonal block indices
        row_start, row_end, col_start, col_end = get_index_upper_M(index * h + j, block_size)
        for r in range(row_start, row_end):
            row_data_start = M.indptr[r]
            row_data_end = M.indptr[r + 1]
            row_columns = M.indices[row_data_start:row_data_end]
            row_values = M.data[row_data_start:row_data_end]
            for i, c in enumerate(row_columns):
                if col_start <= c < col_end:
                    data.append(row_values[i])
                    row.append((r - row_start) + (j - 1) * block_size)
                    col.append((c - col_start) + j * block_size)

        # Lower diagonal block indices
        row_start, row_end, col_start, col_end = get_index_lower_M(index * h + j, block_size)
        for r in range(row_start, row_end):
            row_data_start = M.indptr[r]
            row_data_end = M.indptr[r + 1]
            row_columns = M.indices[row_data_start:row_data_end]
            row_values = M.data[row_data_start:row_data_end]
            for i, c in enumerate(row_columns):
                if col_start <= c < col_end:
                    data.append(row_values[i])
                    row.append((r - row_start) + j * block_size)
                    col.append((c - col_start) + (j - 1) * block_size)

    M = csr_matrix((data, (row, col)), shape=((h+1) * block_size, (h+1) * block_size))

    y0 = np.zeros((h+1) * block_size)
    for j in range(h+1):
        start, end = get_index_f(index * h + j, block_size)
        y0[j*block_size:(j+1)*block_size] = f[start:end]
    if index != p - 1:
        y0[-block_size:] = 0

    y_j = y0
    y_jodd = []

    T_j = []
    A_jinv = []

    Q_j = []

    for i in range(k):
        Q, A, T, S, B, vo, ve = factorize(M, y_j,  block_size)
        Q_j.append(Q)
        A_jinv.append(matrix_block_diagonal_inv(A, block_size))
        T_j.append(T)
        y_jodd.append(vo)

        V = S @ A_jinv[i]
        M = B - V @ T   
        y_j = ve - V @ y_jodd[i]
        

    return Q_j, y_j, y_jodd, T_j, A_jinv, M

def cyclic_reduction_backward_step(x_j, Q_j, y_jodd, T_j, A_jinv, k):
    x_last = x_j
    # Backward pass
    for j in range(k-1, -1, -1):
        x_to_add = np.concatenate((A_jinv[j] @ (y_jodd[j] - T_j[j] @ x_last), x_last))
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

# SCALAR CYCLIC REDUCTION
def scalar_cyclic_reduction(M,f,block_size):
    if not isinstance(M, csr_matrix):
        M = csr_matrix(M)
    
    m,q = M.shape
    assert m == q, "Matrix must be square"

    p = 1
    index = 0

    n = m//block_size - 1
    r = int(np.log2(p))
    k = int(np.log2(n))
    h = int(2**(k - r)) 

    Q_j, y_j, y_jodd, T_j, A_jinv, M_j = cyclic_reduction_forward_step(M, f, p, k-r, h, index, block_size)

    x_k = spsolve(M_j, y_j)

    sol_x = cyclic_reduction_backward_step(x_k, Q_j, y_jodd, T_j, A_jinv, k-r)

    return sol_x

# CYCLIC REDUCTION PARALLEL
def cyclic_redcution_parallel(M, f, p, block_size):
    # p = number of processes
    if not isinstance(M, csr_matrix):
        M = csr_matrix(M)
    if isinstance(f, csr_matrix):
        f = f.toarray()
        if len(f.shape) > 1:
            f = f.flatten()
    if isinstance(f, np.ndarray):
        if len(f.shape) > 1:
            f = f.flatten()

    m,q = M.shape
    assert m == q, "Matrix must be square"
    assert m == f.shape[0], "Matrix and vector must have the same size"

    n = m//block_size - 1
    r = int(np.log2(p))
    k = int(np.log2(n))
    h = int(2**(k - r))

    # FORWARD PASS
    M_k_p = []
    y_j_p = []
    y_jodd_p = []
    T_j_p = []
    A_jinv_p = []
    Q_p = []

    forward_args = [(M, f, p, k-r, h, i, block_size) for i in range(p)]

    if p != 1:
        with Pool(p) as pool:
            forward_results = pool.starmap(cyclic_reduction_forward_step, forward_args)

            for Q_j, y_j, y_jodd, T_j, A_jinv, M_j in forward_results:
                Q_p.append(Q_j)
                y_jodd_p.append(y_jodd)
                T_j_p.append(T_j)
                A_jinv_p.append(A_jinv)
                M_k_p.append(M_j)
                y_j_p.append(y_j)

            M_k = block_diag(M_k_p,format='csr')
            y_k = np.concatenate(y_j_p)

            # 3.10
            Z_p = create_Zp_matrix(p,block_size)  

            M_k = Z_p @ M_k @ Z_p.T
            y_k = Z_p @ y_k
            
            # 3.11
            x_k = spsolve(M_k, y_k) 

            # 3.12
            x_k_list = []
            for i in range (0,p):
                index_1 = get_index_f(i,block_size)[0]
                index_2 = get_index_f(i+1,block_size)[1]
                x_k_list.append(x_k[index_1:index_2])

            # BACKWARD PASS
            x_p = []
            backward_args = [(x_k_list[i], Q_p[i], y_jodd_p[i], T_j_p[i], A_jinv_p[i], k-r) for i in range(p)]
            backward_results = pool.starmap(cyclic_reduction_backward_step, backward_args)
            for sol in backward_results:
                x_p.append(sol)
    else:
        return scalar_cyclic_reduction(M,f,block_size)
    
    # 3.13
    x = np.array([])
    for i in range(p):
        if i == 0:
            x = np.concatenate((x, x_p[i]))
        else:
            x = np.concatenate((x, x_p[i][block_size:]))
    return x


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    # Load harmonic oscillator tests. 
    np.set_printoptions(precision=2, suppress=True)
    # TESTS! 2 PROCESSES
    block_size = 4
    #Ns = [17,33,129,257,513,1025,2049,4097,8193]
    Ns = [16385,32769,65537,131073,262145,524289]
    #processes = [2,4,8]
    Ns = [8193]
    processes = [8]
    resses = []

    for n in Ns:
        for p in processes:
        # res = []
            print(f"Number of processes p: {p}, N: {n}")
            A, f, x = load_npz(f"sparse_harmonic_new/A_{n}.npz"), load_npz(f"sparse_harmonic_new/f_{n}.npz"), load_npz(f"sparse_harmonic_new/x_{n}.npz")
            #A, f, x = load_npz(f"sparse_harmonic/A_test2.npz"), load_npz(f"sparse_harmonic/f_test2.npz"), load_npz(f"sparse_harmonic/x_test2.npz")
            print("Sizes A, f, u: ", A.shape, f.shape, x.shape)


            start = time.time()
            sol = spsolve(A,f)
            end = time.time()
            print(f"Time spsolve: ", end-start,"\n")
           
            start = time.time()
            sol = cyclic_redcution_parallel(A,f,p,block_size)
            end = time.time()
            # thetime = end-start
            # res.append(thetime)
            print(f"Time p={p}: ", end-start)


            print(f"Error: ", np.linalg.norm(x.toarray()-sol.T),"\n")


            #print("Real solution: ", x.toarray().flatten(),"\n") 
            #print("Computed solution: ", sol.flatten())
        # resses.append(res)

    # plt.figure()
    # for i in range(len(processes)):
    #     plt.plot(Ns, resses[i], label=f"p = {processes[i]}")
    # plt.legend(loc="upper right")
    # plt.show()

    # profiler.disable()
    # profiler.print_stats(sort='cumtime')




    

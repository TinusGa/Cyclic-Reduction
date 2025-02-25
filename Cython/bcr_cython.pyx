# cython: boundscheck=False, wraparound=False, nonecheck=False, language_level=3
import numpy as np
cimport numpy as np
from cython.parallel import prange
cimport cython

# We still use some Python libraries for sparse matrices and solvers.
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


##############################################
# Forward reduction – rewritten in Cython   #
##############################################
cpdef tuple forward_reduction(object M, np.ndarray f, int block_size, int nbyp, int processor, int processors):
    """
    Perform the forward step of the Block Cyclic Reduction.
    """
    cdef int start_row = block_size * (nbyp * processor + 1)
    cdef int end_row   = block_size * (nbyp * (processor + 1) + 1)
    cdef list M_current = []
    cdef list f_current = []
    cdef int j

    # Loop over the assigned rows and extract the dense blocks.
    for j in range(start_row, end_row, block_size):
        # Note: These calls still use the Python API (slicing and .toarray())
        B1 = M[j:j+block_size, j-block_size:j].toarray()
        A1 = M[j:j+block_size, j:j+block_size].toarray()
        M_current.append([B1, A1])
        f_current.append(f[j:j+block_size])

    cdef list B_s = []
    cdef list A_s = []
    cdef list f_s = []
    cdef int m_len = len(M_current)

    # Predeclare variables to use in the loop.
    cdef list M_next, f_next
    cdef np.ndarray I = np.eye(block_size)
    cdef int i

    # Declare variables for the inner loop.
    cdef list row_1, row_2
    cdef np.ndarray B1_local, A1_local, B2, A2, f1_local, f2
    cdef np.ndarray A1_inv, B2A1_inv, new_B1, new_A1, new_f1

    # Continue reducing until only one block remains.
    while True:
        m_len = len(M_current)
        if m_len == 1:
            break

        M_next = []
        f_next = []
        # I remains constant as block_size is constant.
        for i in range(0, m_len, 2):
            # Extract blocks for the pair.
            row_1 = M_current[i]
            row_2 = M_current[i+1]
            B1_local = row_1[0]
            A1_local = row_1[1]
            B2 = row_2[0]
            A2 = row_2[1]
            f1_local = f_current[i]
            f2 = f_current[i+1]

            # Save blocks for use in backsubstitution.
            B_s.append(B1_local)
            A_s.append(A1_local)
            f_s.append(f1_local)

            # Compute new blocks.
            A1_inv = np.linalg.solve(A1_local, I)
            B2A1_inv = B2 @ A1_inv
            new_B1 = B2A1_inv @ B1_local
            new_A1 = -A2
            new_f1 = B2A1_inv @ f1_local - f2

            M_next.append([new_B1, new_A1])
            f_next.append(new_f1.flatten())
        M_current = M_next
        f_current = f_next

    return M_current[0], f_current[0], B_s, A_s, f_s



##############################################
# Backsubstitution – rewritten in Cython      #
##############################################
cpdef np.ndarray backsubstitution(list B_s, list A_s, list f_s, list x_s, int number_of_steps, int block_size, int processors):
    """
    Perform the backward step of the Block Cyclic Reduction.
    """
    cdef np.ndarray x_result = x_s[0]
    cdef int power = 0
    cdef int i, idx, num_step_blocks, m_total, num_blocks, j, k
    cdef list B_blocks, A_blocks, f_blocks, x_for_solve_blocks, x_new_list
    cdef np.ndarray x_for_solve, rhs, x_new
    cdef np.ndarray even_blocks, odd_blocks, forward_perm, inv_perm, new_indices

    for i in range(number_of_steps - 1, -1, -1):
        if power > 0:
            j = -(int(2**power) - 1)
        else:
            j = 0  # Equivalent to slicing from index k to the end.
        k = -(int(2**(power+1)) - 1)

        # Use wraparound for negative indices on Python lists.
        with cython.wraparound(True):
            if j == 0:
                B_blocks = B_s[k:]
                A_blocks = A_s[k:]
                f_blocks = f_s[k:]
            else:
                B_blocks = B_s[k:j]
                A_blocks = A_s[k:j]
                f_blocks = f_s[k:j]

        num_step_blocks = len(A_blocks)
        if num_step_blocks == 0:
            break

        x_for_solve = x_result.copy()
        x_for_solve_blocks = []
        for idx in range(num_step_blocks):
            x_for_solve_blocks.append(x_for_solve[idx * block_size : (idx + 1) * block_size])

        x_new_list = []
        for idx in range(num_step_blocks):
            rhs = f_blocks[idx].flatten() - B_blocks[idx] @ x_for_solve_blocks[idx]
            x_new = np.linalg.solve(A_blocks[idx], rhs)
            x_new_list.append(x_new)

        x_new = np.concatenate(x_new_list)
        x_result = np.concatenate((x_result, x_new))

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




##############################################
# BCR – main driver rewritten in Cython       #
##############################################
def BCR_cython(object M, np.ndarray f, int block_size, int processors):
    """
    Performs Block Cyclic Reduction (BCR) in parallel for solving lower block bidiagonal systems.
    Note: Some parts are still using Python-level parallelism. You can later experiment with
    Cython’s prange (with nogil) if the inner work is fully C-level.
    """
    cdef int N = M.shape[0]
    cdef int L = M.shape[1]
    if N != L:
        raise ValueError(f"M must be square but has dimensions {N}x{L}")
    cdef int n = (N - 1) // block_size
    if n % processors != 0:
        raise ValueError("M must have size (n+1)*block_size x (n+1)*block_size, where n = p * 2**k.")
    cdef int nbyp = n // processors
    # Check that nbyp is a power of two.
    if (nbyp & (nbyp - 1)) != 0 or nbyp == 0:
        raise ValueError("n/p is not a power of two.")
    cdef int number_of_steps = int(np.log2(nbyp))
    
    cdef list M_k_list = []
    cdef list f_k_list = []
    cdef list B_k_s_list = []
    cdef list A_k_s_list = []
    cdef list f_k_s_list = []
    cdef int proc

    # Instead of using multiprocessing.Pool, here we simply loop over processors.
    # (You can later try to parallelize this loop using prange if the inner work is C-level.)
    cdef list forward_reduction_results = []
    for proc in range(processors):
        forward_reduction_results.append(
            forward_reduction(M, f, block_size, nbyp, proc, processors)
        )

    for res in forward_reduction_results:
        M_k_list.append(res[0])
        f_k_list.append(res[1])
        B_k_s_list.append(res[2])
        A_k_s_list.append(res[3])
        f_k_s_list.append(res[4])

    # Solve the base case (this part is serial)
    cdef np.ndarray x0 = spsolve(M[:block_size, :block_size], f[:block_size])
    cdef list base_case_x = [x0]
    cdef np.ndarray B_k, A_k, f_k, x_next
    for proc in range(processors):
        # Each M_k_list element is [B, A]
        B_k = M_k_list[proc][0]
        A_k = M_k_list[proc][1]
        f_k = f_k_list[proc]
        x_next = np.linalg.solve(A_k, f_k.flatten() - B_k @ base_case_x[proc])
        base_case_x.append(x_next)

    #cdef np.ndarray final_x = np.array([])
    final_x = np.empty(0)
    cdef list backsubstitution_results = []
    for proc in range(processors):
        backsubstitution_results.append(
            backsubstitution(B_k_s_list[proc], A_k_s_list[proc], f_k_s_list[proc],
                             base_case_x[proc:proc+2], number_of_steps, block_size, processors)
        )
    for x in backsubstitution_results:
        final_x = np.concatenate((final_x, x))

    with cython.wraparound(True):
        final_x = np.concatenate((final_x, base_case_x[-1]))
    return final_x

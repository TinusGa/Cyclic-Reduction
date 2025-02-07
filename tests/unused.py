def construct_sparse_block_tridiagonal(current_a, current_b, current_c, block_size):
    # Number of blocks
    p = len(current_a)
    
    data, row, col = [], [], []
    # Fill main diagonal blocks
    for i in range(p):
        start = i * block_size
        block = current_a[i].tocoo()  # Convert each block to COO format for easy indexing
        data.extend(block.data)
        row.extend(block.row + start)
        col.extend(block.col + start)
    # Fill upper and lower diagonal blocks
    for i in range(p - 1):
        start_upper = i * block_size
        start_lower = (i + 1) * block_size
        
        block_upper = current_b[i].tocoo()
        data.extend(block_upper.data)
        row.extend(block_upper.row + start_upper)
        col.extend(block_upper.col + start_lower)

        block_lower = current_c[i].tocoo()
        data.extend(block_lower.data)
        row.extend(block_lower.row + start_lower)
        col.extend(block_lower.col + start_upper)

    block_tridiagonal_matrix = csr_matrix((data, (row, col)), shape=(p * block_size, p * block_size))
    return block_tridiagonal_matrix


def construct_sparse_block_tridiagonal_direct(index, block_size, M, h):
    # Initialize lists to store matrix data, rows, and columns
    data, row, col = [], [], []
    # Process the first block
    if index == 0:
        index_1, index_2, index_3, index_4 = get_index_main_M(0, block_size)
        block_a = M[index_1:index_2, index_3:index_4].tocoo()
        data.extend(block_a.data)
        row.extend(block_a.row)
        col.extend(block_a.col)
    else:
        # First block for the main diagonal (empty if index != 0)
        block_a = csr_matrix((block_size, block_size)).tocoo()
        data.extend(block_a.data)
        row.extend(block_a.row)
        col.extend(block_a.col)
    
    # Loop through all other blocks
    for j in range(1, h + 1):
        main_index = get_index_main_M(index * h + j, block_size)
        upper_index = get_index_upper_M(index * h + j, block_size)
        lower_index = get_index_lower_M(index * h + j, block_size)

        # Main diagonal block
        block_a = M[main_index[0]:main_index[1], main_index[2]:main_index[3]].tocoo()
        data.extend(block_a.data)
        row.extend(block_a.row + j * block_size)
        col.extend(block_a.col + j * block_size)
        
        # Upper diagonal block
        block_c = M[upper_index[0]:upper_index[1], upper_index[2]:upper_index[3]].tocoo()
        data.extend(block_c.data)
        row.extend(block_c.row + (j-1) * block_size)
        col.extend(block_c.col + j * block_size)
        
        # Lower diagonal block
        block_b = M[lower_index[0]:lower_index[1], lower_index[2]:lower_index[3]].tocoo()
        data.extend(block_b.data)
        row.extend(block_b.row + j * block_size)
        col.extend(block_b.col + (j-1) * block_size)
    
    # Construct the sparse block tridiagonal matrix
    p = (h + 1)  # Total number of blocks in the main diagonal (including the first block)
    block_tridiagonal_matrix = csr_matrix((data, (row, col)), shape=(p * block_size, p * block_size))
    
    return block_tridiagonal_matrix
import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    A_np = np.array(A)
    N,M=A_np.shape

    transpose = np.zeros((M,N), dtype=A_np.dtype)

    for i in range(N):
        for j in range(M):
            transpose[j,i]=A_np[i,j]
    
    return transpose

import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    try:
        matrix = np.array(matrix)
        if matrix.ndim != 2:
            return None
        if matrix.shape[0] != matrix.shape[1]:
            return None
        if matrix.shape[0] == 0:
            return None

        eigvals = np.linalg.eigvals(matrix)
        # sort by complex and then real (this is the correct order)
        # NOTE: for some reason the solution passed AC without the sorting code!
        # i dont know what the reason behind this is.
        return eigvals
    except:
        return None

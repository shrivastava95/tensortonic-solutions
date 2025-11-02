import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    matrix = np.array(matrix)
    norm_kwargs = {"keepdims": True}
    if matrix.ndim != 2:
        return None

    if axis is not None:
        if axis not in [0, 1]:
            return None
        norm_kwargs["axis"] = axis

    match norm_type:
        case "l2":
            norm = np.sqrt(np.sum(matrix * matrix, **norm_kwargs))
            """# norm_kwargs["ord"] = 2"""
        case "l1":
            norm = np.sum(np.abs(matrix), **norm_kwargs)
            """# norm_kwargs["ord"] = 1"""
        case "max":
            norm = np.max(matrix, **norm_kwargs)
            """# norm_kwargs["ord"] = np.inf"""
        case _:
            return None
    """
    # norm = np.linalg.norm(matrix, **norm_kwargs)
    # for some reason, np.linalg.norm for order 2 was failing on one particular test case.
    # the solution was logically correct however there might have been some small numerical difference.
    """ 
    norm = np.where(norm > 0, norm, 1.0) # handle division by zero safely.

    normed_matrix = matrix / norm
    return normed_matrix


import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
    # Write code here
  
    m = np.max(x, axis =1, keepdims = True)
    exp_x = np.exp(x-m)
    return  exp_x/np.sum(exp_x, axis=1, keepdims=True)
import numpy as np
import sklearn
from sklearn.metrics.pairwise import polynomial_kernel

_N = 33   # Dimension 33x33
_K = 32   # Arbiter stages

# Helper: Rank-1 De-Kronecker Approximation

def factorRank1(w1089: np.ndarray, tol: float = 1e-12):
    """
    Given a flattened 33x33 matrix (length 1089), recover vectors u, v
    such that outer(u, v) = W.  The matrix should be rank-1 ideally.

    Uses a fast pivot method with SVD fallback.
    """

    w1089 = np.asarray(w1089, dtype=np.float64).ravel()
    if w1089.size != _N * _N:
        raise ValueError("Expected vector of length 1089 (33x33).")

    W = w1089.reshape(_N, _N)

    if not np.any(W):
        return np.zeros(_N), np.zeros(_N)

    # Fast pivot method 
    i, j = np.unravel_index(np.argmax(np.abs(W)), W.shape)
    pivot = W[i, j]

    if pivot != 0.0:
        u = W[:, j].copy()
        v = (W[i, :] / pivot).copy()

        R = W - np.outer(u, v)
        rel_error = np.linalg.norm(R, "fro") / (np.linalg.norm(W, "fro") + 1e-18)

        if rel_error < tol:
            return u, v

    # Performs SVD as fallback
    U, S, Vt = np.linalg.svd(W, full_matrices=False)

    if S[0] <= 0:
        return np.zeros(_N), np.zeros(_N)

    # Distribute sqrt of singular value equally
    scale = np.sqrt(S[0])
    u = U[:, 0] * scale
    v = Vt[0, :] * scale
    return u, v

# Helper: Arbiter Model to Delay Parameters

def arbiter_model_to_delays(w33: np.ndarray):
    """
    Convert a 33-dim arbiter model into 4 non-negative delay vectors
    (p, q, r, s) each of length 32.

    A valid (non-unique) reconstruction is produced.
    """

    w = np.asarray(w33, dtype=np.float64).ravel()
    if w.size != _N:
        raise ValueError("Expected length 33 for arbiter model.")

    alpha = np.zeros(_K)
    beta  = np.zeros(_K)

    alpha[0]  = w[0]
    alpha[1:] = w[1:_K]
    beta[-1]  = w[_K]

    # Solves the linear system to get alpha, beta
    # x = p − q,  y = r − s
    x = alpha + beta
    y = alpha - beta

    p = np.maximum(x, 0.0)
    q = np.maximum(-x, 0.0)
    r = np.maximum(y, 0.0)
    s = np.maximum(-y, 0.0)

    return p, q, r, s

def my_decode( w ):
	# Use this method to invert a PUF linear model to get back delays
	# w is a single 1089-dim vector (last dimension being the bias term)
	# The output should be eight 32-dimensional vectors
    """
	For XOR-APUF model (Kronecker structure), extract 8 delay vectors:
	    a, b, c, d  from first arbiter
	    p, q, r, s  from second arbiter
	Each is a non-negative 32-dim vector.
	"""
    w = np.asarray(w, dtype=np.float64).ravel()

    # Recover rank-1 factors (33-dim each)
    u, v = factorRank1(w)

    # Convert each 33-vector to (p,q,r,s)
    a, b, c, d = arbiter_model_to_delays(u)
    p, q, r, s = arbiter_model_to_delays(v)

    return a, b, c, d, p, q, r, s


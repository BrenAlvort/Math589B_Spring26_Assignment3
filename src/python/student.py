from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable
import numpy as np


@dataclass
class OdeResult:
    t: np.ndarray
    y: np.ndarray
    status: int = 0
    success: bool = True
    message: str = "The solver successfully reached the end of the integration interval."


def solve_continuous_are(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """
    Solve the continuous-time algebraic Riccati equation

        A^T P + P A - P B R^{-1} B^T P + Q = 0

    using the direct Hamiltonian method.
    """
    A = np.array(A, dtype=float, copy=False)
    B = np.array(B, dtype=float, copy=False)
    Q = np.array(Q, dtype=float, copy=False)
    R = np.array(R, dtype=float, copy=False)

    n = A.shape[0]

    if A.ndim != 2 or A.shape[1] != n:
        raise ValueError("A must be square.")
    if Q.shape != (n, n):
        raise ValueError("Q must have the same shape as A.")
    if B.ndim != 2 or B.shape[0] != n:
        raise ValueError("B must have the same number of rows as A.")
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be square.")

    R_inv = np.linalg.inv(R)

    H = np.block([
        [A, -B @ R_inv @ B.T],
        [-Q, -A.T],
    ])

    eigvals, eigvecs = np.linalg.eig(H)

    stable_idx = np.where(np.real(eigvals) < 0.0)[0]
    if stable_idx.size != n:
        raise np.linalg.LinAlgError(
            f"Expected {n} stable eigenvalues, found {stable_idx.size}."
        )

    V = eigvecs[:, stable_idx]
    V1 = V[:n, :]
    V2 = V[n:, :]

    if np.linalg.matrix_rank(V1) < n:
        raise np.linalg.LinAlgError("Stable invariant subspace is singular.")

    P = V2 @ np.linalg.inv(V1)

    P = np.real_if_close(P, tol=1000)
    P = np.asarray(P, dtype=float)
    P = 0.5 * (P + P.T)

    return P


def _rk4_step(
    f: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    h: float,
) -> np.ndarray:
    k1 = np.asarray(f(t, y), dtype=float)
    k2 = np.asarray(f(t + 0.5 * h, y + 0.5 * h * k1), dtype=float)
    k3 = np.asarray(f(t + 0.5 * h, y + 0.5 * h * k2), dtype=float)
    k4 = np.asarray(f(t + h, y + h * k3), dtype=float)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_ivp(
    fun: Callable[[float, np.ndarray], np.ndarray],
    t_span: tuple[float, float],
    y0: np.ndarray,
    t_eval: Iterable[float] | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    args: tuple = (),
    **kwargs,
) -> OdeResult:
    """
    Minimal SciPy-like IVP solver sufficient for this project.

    Uses classical RK4 and returns an object with attributes .t and .y.
    Parameters rtol, atol, and kwargs are accepted for compatibility.
    """
    del rtol, atol, kwargs

    t0 = float(t_span[0])
    tf = float(t_span[1])

    y0 = np.asarray(y0, dtype=float)
    if y0.ndim != 1:
        raise ValueError("y0 must be one-dimensional.")

    if args:
        def f(t: float, y: np.ndarray) -> np.ndarray:
            return np.asarray(fun(t, y, *args), dtype=float)
    else:
        def f(t: float, y: np.ndarray) -> np.ndarray:
            return np.asarray(fun(t, y), dtype=float)

    if t_eval is None:
        t = np.linspace(t0, tf, 1001)
    else:
        t = np.asarray(list(t_eval), dtype=float)
        if t.ndim != 1:
            raise ValueError("t_eval must be one-dimensional.")
        if t.size == 0:
            raise ValueError("t_eval must contain at least one time.")
        if abs(t[0] - t0) > 1e-12 or abs(t[-1] - tf) > 1e-12:
            raise ValueError("t_eval must start at t_span[0] and end at t_span[1].")

    n = y0.size
    m = t.size
    y = np.zeros((n, m), dtype=float)
    y[:, 0] = y0

    for j in range(m - 1):
        h = t[j + 1] - t[j]
        y[:, j + 1] = _rk4_step(f, t[j], y[:, j], h)

    return OdeResult(t=t, y=y)

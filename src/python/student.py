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

    using the direct Hamiltonian invariant-subspace method.
    """
    A = np.array(A, dtype=float, copy=False)
    B = np.array(B, dtype=float, copy=False)
    Q = np.array(Q, dtype=float, copy=False)
    R = np.array(R, dtype=float, copy=False)

    n = A.shape[0]

    if A.ndim != 2 or A.shape != (n, n):
        raise ValueError("A must be square.")
    if Q.ndim != 2 or Q.shape != (n, n):
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

    # The exact solution is real symmetric; clean numerical noise carefully.
    imag_norm = np.max(np.abs(np.imag(P)))
    if imag_norm > 1e-7:
        raise np.linalg.LinAlgError(
            f"Riccati solution has non-negligible imaginary part: {imag_norm:.3e}"
        )

    P = np.real(P)
    P = 0.5 * (P + P.T)
    return P


def _rk4_step(
    f: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    h: float,
) -> np.ndarray:
    """
    One classical RK4 step:
        y_{n+1} = y_n + h/6 (k1 + 2k2 + 2k3 + k4)
    """
    k1 = np.asarray(f(t, y), dtype=float)
    k2 = np.asarray(f(t + 0.5 * h, y + 0.5 * h * k1), dtype=float)
    k3 = np.asarray(f(t + 0.5 * h, y + 0.5 * h * k2), dtype=float)
    k4 = np.asarray(f(t + h, y + h * k3), dtype=float)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _adaptive_rk4_segment(
    f: Callable[[float, np.ndarray], np.ndarray],
    t0: float,
    y0: np.ndarray,
    t1: float,
    rtol: float,
    atol: float,
) -> np.ndarray:
    """
    Integrate from t0 to t1 using adaptive RK4 with step doubling.

    Error estimate:
      one full step of size h  vs  two half-steps of size h/2.
    """
    direction = np.sign(t1 - t0)
    if direction == 0.0:
        return y0.copy()

    interval = abs(t1 - t0)

    # Conservative initial step. This is intentionally smaller than the raw output spacing.
    h = min(interval, 1e-3)
    t = t0
    y = y0.copy()

    while direction * (t1 - t) > 0:
        h = min(h, abs(t1 - t))
        h_signed = direction * h

        y_big = _rk4_step(f, t, y, h_signed)
        y_half = _rk4_step(f, t, y, 0.5 * h_signed)
        y_small = _rk4_step(f, t + 0.5 * h_signed, y_half, 0.5 * h_signed)

        err = y_small - y_big
        scale = atol + rtol * np.maximum(np.abs(y_small), np.abs(y))
        err_norm = np.max(np.abs(err) / scale)

        if err_norm <= 1.0:
            # Accept the more accurate two-half-step value.
            t = t + h_signed
            y = y_small

            # Grow step cautiously.
            if err_norm == 0.0:
                factor = 2.0
            else:
                factor = min(2.0, max(1.2, 0.9 * err_norm ** (-0.2)))
            h = min(interval, h * factor)
        else:
            # Reject and reduce step.
            factor = max(0.1, 0.9 * err_norm ** (-0.2))
            h = h * factor

            if h < 1e-12:
                raise RuntimeError("Adaptive RK4 step size underflow.")

    return y


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
    Minimal SciPy-like IVP solver sufficient for modal_lqr.py.

    Uses classical RK4 with adaptive step doubling between consecutive
    output times. This keeps compatibility with the handout requirement
    while resolving higher-frequency modes much better than one RK4 step
    per output interval.
    """
    del kwargs  # accepted only for compatibility

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

    current_t = t[0]
    current_y = y0.copy()

    for j in range(m - 1):
        next_t = t[j + 1]
        current_y = _adaptive_rk4_segment(
            f,
            current_t,
            current_y,
            next_t,
            rtol=rtol,
            atol=atol,
        )
        current_t = next_t
        y[:, j + 1] = current_y

    return OdeResult(t=t, y=y)

import ipdb
import numpy as np


def main():
    rng = np.random.default_rng(seed=57213)
    np.set_printoptions(precision=4, linewidth=300, suppress=True)

    batch = 32
    nx = 2

    ls_A = rng.standard_normal((batch, nx))
    ls_B = rng.standard_normal((batch, nx))

    A1, resid, rank, s = np.linalg.lstsq(ls_A, ls_B, 1e-8)

    # Try using SVD.
    U, S, Vh = np.linalg.svd(ls_A, full_matrices=False)
    S_inv = 1 / S

    A2 = Vh.T @ np.diag(S_inv) @ (U.T @ ls_B)

    print(A1)
    print(A2)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()

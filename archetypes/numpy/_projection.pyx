# cython: infer_types=True
import numpy as np

cimport cython
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cython cimport floating
from libc.math cimport fmax


def l1_normalize_proj(floating[:,::1] X not None):
    """first non-negative orthant projection, then l1 normalization."""
    cdef:
        floating row_sum
        Py_ssize_t i, j

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = fmax(X[i, j], 1e-8)

    for i in range(X.shape[0]):
        row_sum = 0.
        for j in range(X.shape[1]):
            row_sum += X[i, j]
        for j in range(X.shape[1]):
            X[i, j] /= row_sum

    return

def unit_simplex_proj(floating[:,::1] X not None):
    """Condat's projection onto the unit simplex."""

    cdef:
        Py_ssize_t m = X.shape[0], n = X.shape[1]
        floating *aux = <floating *> PyMem_Malloc(n * sizeof(floating))
        floating *aux0 = aux
        floating tau = 0.
        Py_ssize_t auxlength = 1, auxlengthold = -1, i = 1, j = 0

    for j in range(m):
        auxlength = 1
        auxlengthold = -1
        aux = aux0
        aux[0] = X[j, 0]
        tau = aux[0] - 1.

        for i in range(1, n):
            if X[j, i] > tau:
                aux[auxlength] = X[j, i]
                tau += (aux[auxlength] - tau) / (auxlength - auxlengthold)
                if tau <= X[j, i] - 1.:
                    tau = X[j, i] - 1.
                    auxlengthold = auxlength - 1
                auxlength += 1

        if auxlengthold >= 0:
            auxlengthold += 1
            auxlength -= auxlengthold
            aux += auxlengthold
            while auxlengthold >= 1:
                auxlengthold -= 1
                if aux0[auxlengthold] > tau:
                    aux -= 1
                    aux[0] = aux0[auxlengthold]
                    auxlength += 1
                    tau += (aux[0] - tau) / auxlength
            auxlengthold -= 1

        auxlengthold = auxlength + 1
        while auxlength <= auxlengthold:
            auxlengthold = auxlength - 1
            auxlength = 0
            for i in range(auxlengthold + 1):
                if aux[i] > tau:
                    aux[auxlength] = aux[i]
                    auxlength += 1
                else:
                    tau += (tau - aux[i]) / (auxlengthold - i + auxlength)

        for i in range(n):
            X[j, i] = fmax(X[j, i] - tau, 0.0)

    PyMem_Free(aux0)

    return

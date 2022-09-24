import numpy as np


def furthest_sum(K, noc, random_state):
    """Furthest sum algorithm, to efficiently generate initial seed/archetypes.

    Soruce: https://github.com/ulfaslak/py_pcha/blob/master/py_pcha/furthest_sum.py

    Note: Commonly data is formatted to have shape (examples, dimensions).
    This function takes input and returns output of the transposed shape,
    (dimensions, examples).

    Parameters
    ----------
    K : np.ndarray
        Either a data matrix or a kernel matrix of two dimensions.

    noc : int
        Number of archetypes to extract.

    random_state: np.random_state
        Random generator.
    Output
    ------
    i : List[int]
        The extracted candidate archetypes
    """

    def max_ind_val(v):
        return max(zip(range(len(v)), v), key=lambda x: x[1])

    i_shape, j_shape = K.shape
    i = [int(np.floor(j_shape * random_state.rand()))]
    index = np.array(range(j_shape))
    index[i] = -1
    ind_t = i
    sum_dist = np.zeros((1, j_shape), np.complex128)

    if j_shape > noc * i_shape:
        Kt = K
        Kt2 = np.sum(Kt**2, axis=0)
        for k in range(1, noc + 11):
            if k > noc - 1:
                Kq = np.dot(Kt[:, i[0]], Kt)
                sum_dist -= np.lib.scimath.sqrt(Kt2 - 2 * Kq + Kt2[i[0]])
                index[i[0]] = i[0]
                i = i[1:]
            t = np.where(index != -1)[0]
            Kq = np.dot(Kt[:, ind_t].T, Kt)
            sum_dist += np.lib.scimath.sqrt(Kt2 - 2 * Kq + Kt2[ind_t])
            ind, val = max_ind_val(sum_dist[:, t][0].real)
            ind_t = t[ind]
            i.append(ind_t)
            index[ind_t] = -1
    else:
        if i_shape != j_shape or np.sum(K - K.T) != 0:  # Generate kernel if K not one
            Kt = K
            K = np.dot(Kt.T, Kt)
            K = np.lib.scimath.sqrt(
                np.tile(np.diag(K), (j_shape, 1))
                - 2 * K
                + np.tile(np.mat(np.diag(K)).T, (1, j_shape))
            )

        Kt2 = np.diag(K)  # Horizontal
        for k in range(1, noc + 11):
            if k > noc - 1:
                sum_dist -= np.lib.scimath.sqrt(Kt2 - 2 * K[i[0], :] + Kt2[i[0]])
                index[i[0]] = i[0]
                i = i[1:]
            t = np.where(index != -1)[0]
            sum_dist += np.lib.scimath.sqrt(Kt2 - 2 * K[ind_t, :] + Kt2[ind_t])
            ind, val = max_ind_val(sum_dist[:, t][0].real)
            ind_t = t[ind]
            i.append(ind_t)
            index[ind_t] = -1
    return i

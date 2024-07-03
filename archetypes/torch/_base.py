import torch
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import check_generator_torch
from ._inits import aa_plus_plus, furthest_first, furthest_sum, uniform


class AABase(BaseEstimator, TransformerMixin):
    """
    Archetype Analysis.

    Parameters
    ----------
    n_archetypes : int
        The number of archetypes to compute.
    max_iter : int, default=300
        Maximum number of iterations of the archetype analysis algorithm
        for a single run.
    tol : float, default=1e-4
        Relative tolerance of two consecutive iterations to declare convergence.
    init : str or callable, default='uniform'
        Method used to initialize the archetypes. If a string, it must be one of
        the following: 'uniform', 'furthest_sum', 'furthest_first' or 'aa_plus_plus'.
        See :ref:`initialization-methods`.
    init_kwargs : dict, default=None
        Additional keyword arguments to pass to the initialization method.
    save_init : bool, default=False
        If True, save the initial coefficients in the attribute `B_init_`.
    method: str, default='nnls'
        The optimization method to use for the archetypes and the coefficients.
        It must be one of the following: 'autogd'.
    method_kwargs : dict, default=None
        Additional arguments to pass to the optimization method. See :ref:`optimization-methods`.
    verbose : bool, default=False
        Verbosity mode.
    device : str or torch.device, default=None
        The device to use for the computation. If None, the default device is used.
    generator : int, Generator instance or None, default=None
        Determines random number generation of coefficients. Use an int to make
        the randomness deterministic.

    Attributes
    ----------

    archetypes_ : np.ndarray
        The computed archetypes.
        It has shape `(n_archetypes, n_features)`.
    archetypes_init_ : np.ndarray
        The initial archetypes. It is only available if `save_init=True`.
    similarity_degree_ : np.ndarray
        The similarity degree of each sample to each archetype.
        It has shape `(n_samples, n_archetypes)`.
    archetypes_similarity_degree_ : np.ndarray
        The similarity degree of each archetype to each sample.
        It has shape `(n_archetypes, n_samples)`.
    labels_ : np.ndarray
        The label of each sample. It is the index of the closest archetype.
        It has shape `(n_samples,)`.
    loss_ : list
        The loss at each iteration.

    References
    ----------

    .. [1] "Archetypal Analysis" by Cutler, A. and Breiman, L. (1994).
    .. [2] "Archetypal Analysis for machine learning and data mining"
           by Morup, M. and Hansen, L. K. (2012).
    """

    def __init__(
        self,
        n_archetypes,
        max_iter=300,
        tol=1e-4,
        init="uniform",
        init_kwargs=None,
        save_init=False,
        method="nnls",
        method_kwargs=None,
        verbose=False,
        device=None,
        generator=None,
    ):
        self.n_archetypes = n_archetypes
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.init_kwargs = init_kwargs
        self.save_init = save_init
        self.method = method
        self.method_kwargs = method_kwargs
        self.verbose = verbose
        self.device = device
        self.generator = generator

        # Init attributes to avoid errors
        self.archetypes_ = None
        self.archetypes_init_ = None
        self.similarity_degree_ = None
        self.archetypes_similarity_degree_ = None
        self.labels_ = None
        self.loss_ = None

        self._check_parameters()

    def _check_data(self, X):
        if X.shape[0] < self.n_archetypes:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_archetypes={self.n_archetypes}."
            )

        X = torch.as_tensor(X, device=self.device)
        return X

    def _check_parameters(self):
        # Check if n_archetypes is a positive integer
        if not isinstance(self.n_archetypes, int):
            raise TypeError
        if self.n_archetypes <= 0:
            raise ValueError(f"n_archetypes should be > 0, got {self.n_archetypes} instead.")

        # Check if max_iter is a positive integer
        if not isinstance(self.max_iter, int):
            raise TypeError
        if self.max_iter <= 0:
            raise ValueError(f"max_iter should be > 0, got {self.max_iter} instead.")

        # Check if tol is positive or zero
        if not isinstance(self.tol, (int, float)):
            raise TypeError
        if self.tol < 0:
            raise ValueError(f"tol should be >= 0, got {self.tol} instead.")

        # Check if init is a string or a function
        if not isinstance(self.init, str):
            raise TypeError(
                f"method_init should be a string or a function, got {self.init} instead."
            )

        # If method_init is a string, check if it is one of the predefined methods
        available_methods_init = ["uniform", "furthest_sum", "furthest_first", "aa_plus_plus"]

        if isinstance(self.init, str):
            if self.init not in available_methods_init:
                raise ValueError(
                    f"algorithm_init must be one of {available_methods_init}, "
                    f"got {self.init} instead."
                )
            else:
                if self.init == "uniform":
                    self.init_c_ = uniform
                elif self.init == "furthest_sum":
                    self.init_c_ = furthest_sum
                elif self.init == "furthest_first":
                    self.init_c_ = furthest_first
                elif self.init == "aa_plus_plus":
                    self.init_c_ = aa_plus_plus

        # Check if method_init_kwargs is a dictionary
        if self.init_kwargs is None:
            self.init_kwargs = {}

        if not isinstance(self.init_kwargs, dict):
            raise TypeError(f"init_kwargs should be a dictionary, got {self.init_kwargs} instead.")

        # Check if save_init is a boolean
        if not isinstance(self.save_init, bool):
            raise TypeError(f"save_init should be a boolean, got {self.save_init} instead.")

        # Check if verbose is a boolean
        if not isinstance(self.verbose, bool):
            raise TypeError(f"verbose should be a boolean, got {self.verbose} instead.")

        # Check if generator is an integer, a RandomState instance or None
        self.generator = check_generator_torch(self.generator)

        # check if device is a string or torch.device
        if self.device is not None:
            if isinstance(self.device, str):
                self.device = torch.device(self.device)
            if not isinstance(self.device, (str, torch.device)):
                raise TypeError(
                    f"device should be a string or torch.device, got {self.device} instead."
                )

        # Check method and method_kwargs
        if self.method not in ["autogd"]:
            raise ValueError(f"Invalid optimization method: {self.method}")

        # Check if method_kwargs is a dictionary
        if self.method_kwargs is None:
            self.method_kwargs = {}
        if not isinstance(self.method_kwargs, dict):
            raise TypeError("method_kwargs should be a dictionary.")

    def fit(self, X, y=None, **fit_params):
        """
        Compute Archetype Analysis.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to compute the archetypes.
            It must be noted that the data will be converted to C ordering,
            which will cause a memory copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        pass

    def transform(self, X):
        """
        Transform X to the archetypal space.

        In the new space, each dimension is the distance to the archetypes.
        Note that even if X is sparse, the array returned by `transform` will
        typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_archetypes)
            X transformed in the new space.
        """
        pass

    def fit_transform(self, X, y=None, **fit_params):
        """
        Compute the archetypes and transform X to the archetypal space.

        Equivalent to fit(X).transform(X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_archetypes)
            X transformed in the new space.
        """
        pass


class BiAABase(BaseEstimator, TransformerMixin):
    """
    BiArchetype Analysis.

    Parameters
    ----------
    n_archetypes : tuple
        The number of archetypes to compute.
    max_iter : int, default=300
        Maximum number of iterations of the archetype analysis algorithm
        for a single run.
    tol : float, default=1e-4
        Relative tolerance of two consecutive iterations to declare convergence.
    init : str or callable, default='uniform'
        Method used to initialize the archetypes. If a string, it must be one of
        the following: 'uniform', 'furthest_sum', 'furthest_first' or 'aa_plus_plus'.
        See :ref:`initialization-methods`.
    init_kwargs : dict, default=None
        Additional keyword arguments to pass to the initialization method.
    save_init : bool, default=False
        If True, save the initial coefficients in the attribute `B_init_`.
    method: str, default='nnls'
        The optimization method to use for the archetypes and the coefficients.
        It must be one of the following: 'nnls', 'pgd'.
    method_kwargs : dict, default=None
        Additional arguments to pass to the optimization method. See :ref:`optimization-methods`.
    verbose : bool, default=False
        Verbosity mode.
    device : str or torch.device, default=None
        The device to use for the computation. If None, the default device is used.
    generator : int, RandomState instance or None, default=None
        Determines random number generation of coefficients. Use an int to make
        the randomness deterministic.

    Attributes
    ----------

    archetypes_ : np.ndarray of shape (*n_archetypes)
        The computed archetypes.
    archetypes_init_ : np.ndarray
        The initial archetypes. It is only available if `save_init=True`.
    similarity_degree_ : list[np.ndarray]
        The similarity degree of each sample to each archetype in each dimension.
        Each array has shape `(n_samples, n_archetypes)`.
    archetypes_similarity_degree_ : list[np.ndarray]
        The similarity degree of each archetype to each sample in each dimension.
        Each array has shape `(n_archetypes, n_samples)`.
    labels_ : list[np.ndarray]
        The label of each sample in each dimension. It is the index of the closest archetype.
        Each array has shape `(n_samples,)`.
    loss_ : list
        The loss at each iteration.

    References
    ----------

    .. [1] "Archetypal Analysis" by Cutler, A. and Breiman, L. (1994).
    .. [2] "Biarchetype Analysis: Simultaneous Learning of Observations
           and Features Based on Extremes"
           by Alcacer, A., Epifanio, I. and Gual-Arnau, X. (2024).

    """

    def __init__(
        self,
        n_archetypes,
        max_iter=300,
        tol=1e-4,
        init="uniform",
        init_kwargs=None,
        save_init=False,
        method="nnls",
        method_kwargs=None,
        verbose=False,
        device=None,
        generator=None,
    ):
        self.n_archetypes = n_archetypes
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.init_kwargs = init_kwargs
        self.save_init = save_init
        self.method = method
        self.method_kwargs = method_kwargs
        self.verbose = verbose
        self.device = device
        self.generator = generator

        # Init attributes to avoid errors
        self.A_ = None
        self.B_ = None
        self.archetypes_ = None
        self.archetypes_init_ = None
        self.similarity_degree_ = None
        self.archetypes_similarity_degree_ = None
        self.labels_ = None
        self.loss_ = None

        self._check_parameters()

    def _check_data(self, X):
        if X.shape[0] < self.n_archetypes[0] or X.shape[1] < self.n_archetypes[1]:
            raise ValueError(
                f"n_archetypes should be less than the number of samples and features, "
                f"got {self.n_archetypes} and {X.shape} instead."
            )

        X = torch.as_tensor(X, device=self.device)
        return X

    def _check_parameters(self):
        # Check if n_archetypes is a positive integer
        if not isinstance(self.n_archetypes, tuple):
            raise TypeError
        if len(self.n_archetypes) != 2:
            raise ValueError(
                f"n_archetypes should be a tuple of length 2, got {self.n_archetypes} instead."
            )
        if self.n_archetypes[0] <= 0 or self.n_archetypes[1] <= 0:
            raise ValueError(f"n_archetypes should be > 0, got {self.n_archetypes} instead.")

        # Check if max_iter is a positive integer
        if not isinstance(self.max_iter, int):
            raise TypeError
        if self.max_iter <= 0:
            raise ValueError(f"max_iter should be > 0, got {self.max_iter} instead.")

        # Check if tol is positive or zero
        if not isinstance(self.tol, (int, float)):
            raise TypeError
        if self.tol < 0:
            raise ValueError(f"tol should be >= 0, got {self.tol} instead.")

        # Check if init is a string or a function
        if not isinstance(self.init, str):
            raise TypeError(f"method_init should be a string, got {self.init} instead.")

        # If method_init is a string, check if it is one of the predefined methods
        available_methods_init = ["uniform", "furthest_sum", "furthest_first", "aa_plus_plus"]

        if isinstance(self.init, str):
            if self.init not in available_methods_init:
                raise ValueError(
                    f"algorithm_init must be one of {available_methods_init}, "
                    f"got {self.init} instead."
                )
            else:
                if self.init == "uniform":
                    self.init_c_ = uniform
                elif self.init == "furthest_sum":
                    self.init_c_ = furthest_sum
                elif self.init == "furthest_first":
                    self.init_c_ = furthest_first
                elif self.init == "aa_plus_plus":
                    self.init_c_ = aa_plus_plus

        # Check if method_init_kwargs is a dictionary
        if self.init_kwargs is None:
            self.init_kwargs = {}

        if not isinstance(self.init_kwargs, dict):
            raise TypeError(f"init_kwargs should be a dictionary, got {self.init_kwargs} instead.")

        # Check if save_init is a boolean
        if not isinstance(self.save_init, bool):
            raise TypeError(f"save_init should be a boolean, got {self.save_init} instead.")
        self._algorithm_init = self.init

        # Check if verbose is a boolean
        if not isinstance(self.verbose, bool):
            raise TypeError(f"verbose should be a boolean, got {self.verbose} instead.")

        # Check if generator is an integer, a RandomState instance or None
        self.generator = check_generator_torch(self.generator)

        # check if device is a string or torch.device
        if self.device is not None:
            if isinstance(self.device, str):
                self.device = torch.device(self.device)
            if not isinstance(self.device, (str, torch.device)):
                raise TypeError(
                    f"device should be a string or torch.device, got {self.device} instead."
                )

        # Check method and method_kwargs
        if self.method not in ["autogd"]:
            raise ValueError(f"Invalid optimization method: {self.method}")

        # Check if method_kwargs is a dictionary
        if self.method_kwargs is None:
            self.method_kwargs = {}

        if not isinstance(self.method_kwargs, dict):
            raise TypeError("method_kwargs should be a dictionary.")

    def fit(self, X, y=None, **fit_params):
        """
        Compute Archetype Analysis.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to compute the archetypes.
            It must be noted that the data will be converted to C ordering,
            which will cause a memory copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        pass

    def transform(self, X):
        """
        Transform X to the archetypal space.

        In the new space, each dimension is the distance to the archetypes.
        Note that even if X is sparse, the array returned by `transform` will
        typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_archetypes)
            X transformed in the new space.
        """
        pass

    def fit_transform(self, X, y=None, **fit_params):
        pass

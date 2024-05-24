import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from ._inits import uniform


class AABase(BaseEstimator, TransformerMixin):
    """
    Archetype Analysis.

    Parameters
    ----------
    n_archetypes : int, default=4
        The number of archetypes to compute.
    max_iter : int, default=300
        Maximum number of iterations of the archetype analysis algorithm
        for a single run.
    tol : float, default=1e-4
        Relative tolerance of two consecutive iterations to declare convergence.
    init : str or callable, default='uniform'
        Method used to initialize the archetypes. If a string, it must be one of
        the following: 'uniform', 'furthest_sum', 'furthest_first' or 'aa_plus_plus'.
    init_kwargs : dict, default=None
        Additional keyword arguments to pass to the initialization method.
    save_init : bool, default=False
        If True, save the initial coefficients in the attribute `B_init_`.
    verbose : bool, default=False
        Verbosity mode.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation of coefficients. Use an int to make
        the randomness deterministic.

    Attributes
    ----------

    archetypes_ : np.ndarray of shape (n_archetypes, n_features)
        The computed archetypes.
    similarity_degree_ : np.ndarray of shape (n_samples, n_archetypes)
        The similarity degree of each sample to each archetype.
    archetypes_similarity_degree_ : np.ndarray of shape (n_archetypes, n_samples)
        The similarity degree of each archetype to each sample.
    labels_ : np.ndarray of shape (n_samples,)
        The label of each sample. It is the index of the closest archetype.
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
        n_archetypes=4,
        max_iter=300,
        tol=1e-4,
        init="uniform",
        init_kwargs=None,
        save_init=False,
        verbose=False,
        random_state=None,
    ):
        self.n_archetypes = n_archetypes
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.init_kwargs = init_kwargs
        self.save_init = save_init
        self.verbose = verbose
        self.random_state = random_state

    def _check_data(self, X):
        if X.shape[0] < self.n_archetypes:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_archetypes={self.n_archetypes}."
            )

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
        if not callable(self.init) and not isinstance(self.init, str):
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
                self.init = eval(self.init)

        # If method_init is a function, check if it has the right arguments
        args_names = ["X", "k", "random_state"]
        if callable(self.init) and not all(
            arg in self.init.__code__.co_varnames for arg in args_names
        ):
            raise ValueError(f"method_init should have the following arguments: {args_names}")

        # Check if method_init_kwargs is a dictionary
        if self.init_kwargs is not None and not isinstance(self.init_kwargs, dict):
            raise TypeError(
                f"method_init_kwargs should be a dictionary, got {self.init_kwargs} instead."
            )
        if self.init_kwargs is None:
            self.init_kwargs = {}

        # Check if save_init is a boolean
        if not isinstance(self.save_init, bool):
            raise TypeError(f"save_init should be a boolean, got {self.save_init} instead.")
        self._algorithm_init = self.init

        # Check if verbose is a boolean
        if not isinstance(self.verbose, bool):
            raise TypeError(f"verbose should be a boolean, got {self.verbose} instead.")

        # Check if random_state is an integer, a RandomState instance or None
        self.random_state = check_random_state(self.random_state)

    def _init_B(self, X):
        B = np.zeros((self.n_archetypes, X.shape[0]), dtype=np.float64)

        ind = self.init(
            X, self.n_archetypes, random_state=self.random_state, kwargs=self.init_kwargs
        )

        for i, j in enumerate(ind):
            B[i, j] = 1

        self.B_ = B

    def _init_A(self, X):
        # TODO: Improve it?
        A = np.zeros((X.shape[0], self.n_archetypes), dtype=np.float64)

        ind = uniform(X, self.n_archetypes, random_state=self.random_state, kwargs=self.init_kwargs)

        for i, j in enumerate(ind):
            A[j, i] = 1

        self.A_ = A

    def _optim_A(self, X):
        pass

    def _optim_B(self, X):
        pass

    def _loss(self, X):
        return np.linalg.norm(X - self.A_ @ self.archetypes_) ** 2

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
        X = self._validate_data(X, dtype=[np.float64, np.float32])
        self._check_parameters()
        self._check_data(X)

        # Initialize coefficients
        self._init_A(X)  # Initialize A uniformly
        self._init_B(X)
        if self.save_init:
            self.B_init_ = self.B_.copy()

        self.archetypes_ = self.B_ @ X

        rss = self._loss(X)
        self.loss_ = [rss]

        for i in range(self.max_iter):
            # Verbose mode (print RSS)
            if self.verbose and i % 10 == 0:
                print(f"Iteration {i}/{self.max_iter}: RSS = {rss}")

            # Optimize coefficients
            self._optim_A(X)
            self._optim_B(X)

            self.archetypes_ = self.B_ @ X

            # Compute RSS
            rss = self._loss(X)
            self.loss_.append(rss)
            if abs(self.loss_[-1] - self.loss_[-2]) < self.tol:
                break

        # Set attributes
        self.similarity_degree_ = self.A_
        self.archetypes_similarity_degree_ = self.B_
        self.labels_ = np.argmax(self.A_, axis=1)

        return self

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
        check_is_fitted(self)
        X = self._validate_data(X, dtype=[np.float64, np.float32])

        return self._optim_A(X)

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
        return self.fit(X, y, **fit_params).transform(X)

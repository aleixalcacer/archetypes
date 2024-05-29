import numpy as np
from custom_inherit import doc_inherit
from sklearn.utils.validation import check_is_fitted

from ._base import BiAABase
from ._biaa_3 import BiAA_3


@doc_inherit(parent=BiAABase, style="numpy_with_merge")
class BiAA(BiAABase):
    """
    Archetype Analysis.
    """

    def __init__(
        self,
        n_archetypes,
        max_iter=300,
        tol=1e-4,
        init="uniform",
        init_kwargs=None,
        save_init=False,
        verbose=False,
        random_state=None,
        method="nnls",
        method_kwargs=None,
    ):
        super().__init__(
            n_archetypes=n_archetypes,
            max_iter=max_iter,
            tol=tol,
            init=init,
            init_kwargs=init_kwargs,
            save_init=save_init,
            method=method,
            method_kwargs=method_kwargs,
            verbose=verbose,
            random_state=random_state,
        )

        # Check params for the optimization method
        if self.method in ["nnls", "pgd", "jax"]:
            self.method_class = BiAA_3(
                n_archetypes=self.n_archetypes,
                max_iter=self.max_iter,
                tol=self.tol,
                init=self.init,
                init_kwargs=self.init_kwargs,
                save_init=self.save_init,
                method=self.method,
                method_kwargs=self.method_kwargs,
                verbose=self.verbose,
                random_state=self.random_state,
            )

    def fit(self, X, y=None, **fit_params):
        X = self._validate_data(X, dtype=[np.float64, np.float32])
        self._check_data(X)

        self.method_class.fit(X)

        # Copy attributes
        self.archetypes_ = self.method_class.archetypes_
        self.archetypes_init_ = self.method_class.archetypes_init_
        self.similarity_degree_ = self.method_class.similarity_degree_
        self.archetypes_similarity_degree_ = self.method_class.archetypes_similarity_degree_
        self.labels_ = self.method_class.labels_
        self.loss_ = self.method_class.loss_

        return self

    def transform(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, dtype=[np.float64, np.float32])

        return self.method_class.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

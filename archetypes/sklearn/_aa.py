import optax
from custom_inherit import doc_inherit

from ._aabase import AABase
from ._optims import AAOptimizer, jax_optimizer, nnls_optimizer, pgd_optimizer

optax


@doc_inherit(parent=AABase, style="numpy_with_merge")
class AA(AABase):
    """
    Parameters
    ----------
    method: str, default='nnls'
        The optimization method to use for the archetypes and the coefficients.
    method_kwargs : dict, default=None
        Additional keyword arguments to pass to the optimization method. See TODO for more details.

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
            verbose=verbose,
            random_state=random_state,
        )

        self.method = method
        self.method_kwargs = method_kwargs

        self._check_parameters_()

    def _check_parameters_(self):
        # Check if method is a valid optimization method
        if self.method not in ["nnls", "pgd", "jax"]:
            raise ValueError(f"Invalid optimization method: {self.method}")

        # Check if method_kwargs is a dictionary
        if self.method_kwargs is None:
            self.method_kwargs = {}
        if self.method_kwargs is not None and not isinstance(self.method_kwargs, dict):
            raise TypeError("method_kwargs should be a dictionary.")

        # Check params for the optimization method
        if self.method == "nnls":
            self.method_c: AAOptimizer = nnls_optimizer  # dataclass
            self.max_iter_optimizer = self.method_kwargs.get("max_iter_optimizer", None)
        elif self.method == "pgd":
            self.method_c: AAOptimizer = pgd_optimizer
            self.beta_ = self.method_kwargs.get("beta", 0.5)
            self.n_iter_optimizer = self.method_kwargs.get("n_iter_optimizer", 10)
            self.max_iter_optimizer = self.method_kwargs.get("max_iter_optimizer", 1_000)
            self.learning_rate = self.method_kwargs.get("learning_rate", 1.0)
            self.step_size_A_ = self.learning_rate
            self.step_size_B_ = self.learning_rate
        elif self.method == "jax":
            self.method_c: AAOptimizer = jax_optimizer
            self.optimizer = self.method_kwargs.get("optimizer", "sgd")
            self.optimizer_c = eval(f"optax.{self.optimizer}")
            self.optimizer_kwargs = self.method_kwargs.get(
                "optimizer_kwargs", {"learning_rate": 1e-3}
            )

    def _init_A(self, X):
        return self.method_c.A_init(self, X)

    def _init_B(self, X):
        return self.method_c.B_init(self, X)

    def _optim_B(self, X):
        self.method_c.B_optimize(self, X)

    def _optim_A(self, X):
        self.method_c.A_optimize(self, X)

    def fit(self, X, y=None, **fit_params):
        return self.method_c.fit(self, X, y, **fit_params)

(optimization-methods)=

# Optimization Methods


## No-Negative Least Squares

| Parameter            | Type   | Description                                                                  |
|----------------------|--------|------------------------------------------------------------------------------|
| `const`              | float  | The constant to add in the nnls optimization to enforce convex optimization. |
| `max_iter_optimizer` | int    | The maximum number of iterations for the optimizer.                          |


## Projected Gradient Descent

| Parameter            | Type  | Description                                         |
|----------------------|-------|-----------------------------------------------------|
| `n_iter_optimizer`   | int   | The number of repetitions of the optimization.      |
| `max_iter_optimizer` | int   | The maximum number of iterations for the optimizer. |
| `beta`               | float | The decay factor for the learning rate.             |

## JAX Optimizers

| Parameter          | Type                                | Description                                                                                                                 |
|--------------------|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `optimizer`        | str or optax.GradientTransformation | The optimization method to use. See the [available optimizers](https://optax.readthedocs.io/en/latest/api/optimizers.html). |
| `optimizer_kwargs` | dict                                | The arguments to pass to initialize the optimization method.                                                                |

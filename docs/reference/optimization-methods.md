(optimization-methods)=

# Optimization Methods


## No-Negative Least Squares

| Parameter            | Type   | Description                                                                  |
|----------------------|--------|------------------------------------------------------------------------------|
| `max_iter_optimizer` | int    | The maximum number of iterations in the nnls optimization.                   |
| `const`              | float  | The constant to add in the nnls optimization to enforce convex optimization. |


## Projected Gradient Descent

| Parameter            | Type  | Description                                                        |
|----------------------|-------|--------------------------------------------------------------------|
| `max_iter_optimizer` | int   | The maximum number of iterations for optimizing the learning rate. |
| `beta`               | float | The decay factor for the learning rate.                            |

## JAX Gradient Descent

| Parameter          | Type                                | Description                                                                                                                 |
|--------------------|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `optimizer`        | str or optax.GradientTransformation | The optimization method to use. See the [available optimizers](https://optax.readthedocs.io/en/latest/api/optimizers.html). |
| `optimizer_kwargs` | dict                                | The arguments to pass to initialize the optimization method.                                                                |

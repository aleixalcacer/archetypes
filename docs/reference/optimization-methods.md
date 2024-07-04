(optimization-methods)=

# Optimization Methods

## Overview

### No-Negative Least Squares

{cite:p}`cutler_archetypal_1994`

### Projected Gradient Descent

{cite:p}`morup_archetypal_2012`

### Automatic Gradient Descent



## Parameters

### Numpy backend

* No-Negative Least Squares (`nnls`):

    | Parameter            | Type   | Description                                                                  |
    |----------------------|--------|------------------------------------------------------------------------------|
    | `max_iter_optimizer` | int    | The maximum number of iterations in the nnls optimization.                   |
    | `const`              | float  | The constant to add in the nnls optimization to enforce convex optimization. |

* Projected Gradient Descent (`pgd`):

    | Parameter            | Type  | Description                                                        |
    |----------------------|-------|--------------------------------------------------------------------|
    | `max_iter_optimizer` | int   | The maximum number of iterations for optimizing the learning rate. |
    | `beta`               | float | The decay factor for the learning rate.                            |

### JAX backend

* Automatic Gradient Descent (`autogd`):

    | Parameter          | Type                                | Description                                                                                                                 |
    |--------------------|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
    | `optimizer`        | str or optax.GradientTransformation | The optimization method to use. See the [available optimizers](https://optax.readthedocs.io/en/latest/api/optimizers.html). |
    | `optimizer_kwargs` | dict                                | The arguments to pass to initialize the optimization method.                                                                |


### Torch backend

* Automatic Gradient Descent (`autogd`)

    | Parameter          | Type                                | Description                                                                                                            |
    |--------------------|-------------------------------------|------------------------------------------------------------------------------------------------------------------------|
    | `optimizer`        | str or torch.optim.Optimizer        | The optimization method to use. See the [available optimizers](https://pytorch.org/docs/stable/optim.html#Algorithms). |
    | `optimizer_kwargs` | dict                                | The arguments to pass to initialize the optimization method.                                                           |

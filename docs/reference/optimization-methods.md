(optimization-methods)=

# Optimization Methods

## Overview

### Non-Negative Least Squares

{cite:p}`cutler_archetypal_1994`

### Pseudo Projected Gradient Descent

{cite:p}`morup_archetypal_2012`

### Projected Graident Descent

Like Pseudo-PGD, but performs orthogonal projection onto unit simplex

{cite:p}`condat2016`

### Automatic Gradient Descent



## Parameters

### Numpy backend

* Non-Negative Least Squares (`nnls`):

    | Parameter            | Type   |  Default   | Description                                                                               |
    |----------------------|--------|------------|-------------------------------------------------------------------------------------------|
    | `max_iter_optimizer` | int    |    100     | The maximum number of iterations in the nnls optimization, passed to `scipy.optimize.nnls`|
    | `const`              | float  |   100.0    | The penalization constant to add in the nnls optimization to enforce convex optimization. |

* (Pseudo) Projected Gradient Descent (`pgd` and `pseudo_pgd`):

    | Parameter            | Type  |  Default   | Description                                                        |
    |----------------------|-------|------------|--------------------------------------------------------------------|
    | `max_iter_optimizer` | int   |     10     | The maximum number of iterations for optimizing the learning rate. |
    | `beta`               | float |    0.5     | The decay factor for the learning rate.                            |
    | `step_size`          | float |    1.0     | The initial learning rate at the beginning of optimization, corresponding to the objective function $\frac{1}{2} \left\lVert X - A B X \right \rVert^2$                  |

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

# Archetypes
![PyPI](https://img.shields.io/pypi/v/archetypes)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/archetypes)
[![Python package](https://github.com/aleixalcacer/archetypes/actions/workflows/python-package.yml/badge.svg)](https://github.com/aleixalcacer/archetypes/actions/workflows/python-package.yml)
![PyPI - License](https://img.shields.io/pypi/l/archetypes)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](CODE_OF_CONDUCT.md)

**archetypes** is a [scikit-learn](https://scikit-learn.org) compatible Python package for archetypal analysis.


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install archetypes.

```bash
pip install archetypes
```


## Usage

```python
import archetypes as arch
import numpy as np

X = np.random.normal(0, 1, (100, 2))

aa = arch.AA(n_archetypes=4)

X_trans = aa.fit_transform(X)

```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


## License

Distributed under the BSD 3-Clause License. See [LICENSE](LICENSE) for more information.

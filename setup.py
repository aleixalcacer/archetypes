from setuptools import setup
import os


ver_file = os.path.join('archetypes', 'version.py')
with open(ver_file) as f:
    exec(f.read())

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Natural Language :: English
Operating System :: OS Independent
Programming Language :: Python
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Topic :: Scientific/Engineering
"""

setup(
    name="archetypes",
    description="A scikit-learn compatible Python package for archetypal analysis.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="BSD (3-clause)",
    version=__version__,
    author="Aleix Alcacer",
    author_email="aleixalcacer@gmail.com",
    url="https://github.com/aleixalcacer/archetypes",
    packages=["archetypes"],
    install_requires=["scikit-learn", "numpy", "scipy"],
    python_requires=">=3.7",
    classifiers=list(filter(None, CLASSIFIERS.split("\n"))),
)

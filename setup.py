from setuptools import setup


CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Environment :: Console
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Natural Language :: English
Operating System :: OS Independent
Programming Language :: Python
Topic :: Software Development :: Libraries :: Python Modules
"""


setup(
    name="archetypes",
    description="A scikit-learn compatible Python package for archetypal analysis.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="BSD (3-clause)",
    version="0.0.2",
    author="Aleix Alcacer",
    author_email="aleixalcacer@gmail.com",
    url="https://github.com/aleixalcacer/archetypes",
    packages=["archetypes"],
    install_requires=["scikit-learn", "numpy", "scipy"],
    python_requires=">=3.7",
    classifiers=list(filter(None, CLASSIFIERS.split("\n"))),
)

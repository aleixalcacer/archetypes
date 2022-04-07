# Release guide

```{admonition} This document is only relevant for Archetypes release managers.
:class: seealso

A guide for developers who are doing a Archetypes release.
```


1. Check that `archetypes/version.py` contains the correct version number.

2. Commit the changes:
  
        git commit -a -m "Getting ready for release X.Y.Z"
        git push
  
3. Create a tag `X.Y.Z` from `main` and push it to the github repo.
  Use the next message:

        git tag -a vX.Y.Z -m "Tagging version X.Y.Z"
        git push --tags

4. [Create a release](https://github.com/aleixalcacer/archetypes/releases) on Github.
  
5. Edit the version number in `archetypes/version.py` to increment the version
  to the next minor one (i.e. `X.Y.Z` -> `X.Y.(Z+1).dev0`).

6. Commit your changes with:

        git commit -a -m "Post X.Y.Z release actions done"
        git push

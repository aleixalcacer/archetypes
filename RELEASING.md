# Releasing

* Check that `archetypes/_version.py` contains the correct version number.

* Commit the changes:
  
        git commit -a -m "Getting ready for release X.Y.Z"
        git push
  
* Create a tag `X.Y.Z` from `main` and push it to the github repo.
  Use the next message:

        git tag -a vX.Y.Z -m "Tagging version X.Y.Z"
        git push --tags

* Create a release on https://github.com/aleixalcacer/archetypes/releases.
  
* Edit the version number in `archetypes/_version.py` to increment the version
  to the next minor one (i.e. `X.Y.Z` -> `X.Y.(Z+1).dev0`).

* Commit your changes with:

        git commit -a -m "Post X.Y.Z release actions done"
        git push

# Release guide

```{admonition} This document is only relevant for Archetypes release managers.
:class: seealso

A guide for developers who are doing a Archetypes release.
```


1. Bump the version of the project using a valid bump rule (`patch`, `minor`, `major`)
  according to the release commits:

        hatch version [bump rule]

2. Commit the changes:

        git commit -a -m "Getting ready for release $(hatch version)"
        git push

3. Create a tag from `main` and push it to the Github repo. Use the next message:

        git tag -a v$(hatch version) -m "Tagging version $(hatch version)"
        git push --tags

4. [Create a release](https://github.com/aleixalcacer/archetypes/releases) on Github.

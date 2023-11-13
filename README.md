<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/src/assets/logo-horizontal.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/src/assets/logo-horizontal-dark.svg">
  <img alt="Thunderbolt.jl logo." src="docs/src/assets/logo-horizontal.svg">
</picture>

A modular high-performance framework for cardiac multiphysics.

!!! warning
    This package is under heavy development. Expect regular breaking changes
    for now. If you are interested in joining development, then either comment
    an issue or reach out via julialang.zulipchat.com, via mail or via 
    julialang.slack.com. Alternatively open a discussion if you have something 
    specific in mind.

## Accessing the documentation

Currently the documentation can only be built and viewed locally. Note that the
documentation is incomplete at this point.

To build the docs you first need to install Julia, see <https://julialang.org/> for details.
Then you need to install the packages. To accomplish this you first start a julia shell in the 
docs fonder and open the Pkg REPL; i.e. press `]` at the `julia>` promp to
enter `pkg>` mode. Then you [activate and instantiate](https://pkgdocs.julialang.org/v1/environments/#Using-someone-else's-project) via
```
(@v1.9) pkg> activate .
(docs) pkg> instantiate
```
This has to be done only once. Now to build the docs you can execute the makefile by opening a terminal in the 
docs folder and typing
```
$ julia --project=. make.jl
```
This creates a folder called "build" in the docs folder, which contains the html documentation.
Now just open the contained "index.html" with your browser. All further information will be provided there.

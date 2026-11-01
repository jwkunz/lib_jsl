# Introduction

This book is a companion to [lib_jsl](https://github.com/jwkunz/lib_jsl), a Rust library of
algorithms for scientific computing: numerical integration and differential equations,
optimisation, special functions, signal processing, electromagnetics, statistics, machine
learning and more. It is written for advanced undergraduates who want to understand **how these
algorithms work and why**, not only how to call them.

Each algorithm has its own entry, written like an encyclopedia article. An entry explains what
the algorithm computes and when to use it, where it came from, the mathematical idea that makes
it work (with proofs where they are short enough to be illuminating), how the library implements
it, and where to read more. Figures and tables are produced by running the library itself, so
every number you see is one the code actually computes.

## Two kinds of documentation

| | This book | The API reference |
|---|---|---|
| Purpose | Understand the algorithms | Use the library |
| Content | Motivation, history, derivations, figures, exercises | Every type, function and parameter |
| Organisation | By topic, from simple to advanced | By crate and module |
| Link | You are here | [API reference](../api/lib_jsl/index.html) |

Entries link to the exact API items they describe, and the API documentation links back here.

## Prerequisites

Calculus, linear algebra and some familiarity with Taylor series are assumed throughout.
Chapters build on one another loosely: the integration chapter, for example, uses polynomial
interpolation and orthogonal polynomials, which are introduced where they are needed. The code
is Rust, but the excerpts are short and annotated; readers who know C, Python or MATLAB will be
able to follow them.

## Running the examples

Every entry ends with a runnable example from the repository. To run one:

```console
git clone https://github.com/jwkunz/lib_jsl.git
cd lib_jsl
cargo run --release -p lib_jsl_integration --example quadrature_gauss_legendre
```

The library is pure Rust with no system dependencies, so a Rust toolchain is all you need.

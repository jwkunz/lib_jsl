# lib_jsl

"Jake's Scientific Library" is a compendium of scientific computing routines written in Rust.  This library was originally developed by the author during the course of his PhD work in C++, but has now been rewritten in the Rust language.  Some of the content and organization is inspired by the famous book "Numerical Recipes: The Art of Scientific Computing", but there are many other routines taken from other sources. 

## Note on Style

Although some of routines are translations of routines and algorithms originally authored in C/C++ (fast,optimized,terse), in all cases the author has attempted to use good style and adhere to the principles of idiomatic Rust.  The reader should find the documentation in the codebase thoughtful and helpful, but not a full tutorial on the code or the algorithms presented.

## Scope and Dependencies

The library has minimalistic dependencies, but does use the quasi-standard packages: num and ndarray.  This library acknowledges the years of effort and talent in creating high performance linear algebra and fast fourier transform libraries, and relies on those implementations where appropriate. 

## Sections
The library is organized by high level sections:
*  Interpolation

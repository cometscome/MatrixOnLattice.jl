# MatrixOnLattice

[![Build Status](https://github.com/cometscome/MatrixOnLattice.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cometscome/MatrixOnLattice.jl/actions/workflows/CI.yml?query=branch%3Amain)


# example

```julia
using MatrixOnLattice
using Test
using LinearAlgebra

function test()
    NX = 16
    NY = 16
    NZ = 16
    NT = 16
    NC = 3

    U = MatrixOnLattice4D(NC, NX, NY, NZ, NT)
    U1 = Randomfield(NC, NX, NY, NZ, NT)
    U2 = Identityfield(NC, NX, NY, NZ, NT)
    U3 = Identityfield(NC, NX, NY, NZ, NT)

    mul!(U, U2, U3)
    val = tr(U)
    println(val)
end
test()
```
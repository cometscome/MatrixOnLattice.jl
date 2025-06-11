# MatrixOnLattice

[![Build Status](https://github.com/cometscome/MatrixOnLattice.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cometscome/MatrixOnLattice.jl/actions/workflows/CI.yml?query=branch%3Amain)


# example

```julia
using MatrixOnLattice
using Test
using LinearAlgebra
import CUDA

function test()
    NX = 16
    NY = 16
    NZ = 16
    NT = 16
    NC = 3

    U = MatrixOnLattice4D(NC, NX, NY, NZ, NT)
    U1 = Randomfield(NC, NX, NY, NZ, NT)
    U2 = Randomfield(NC, NX, NY, NZ, NT)
    U3 = Randomfield(NC, NX, NY, NZ, NT)

    substitute!(U,U1)
    mul!(U, U2, U3)
    @time mul!(U, U2, U3)
    @time mul!(U, U2, U3)

    val = tr(U)
    println("cpu: ",val)

    Ug = MatrixOnLattice4D(NC, NX, NY, NZ, NT; accelarator="cuda")
    Ug1 = MatrixOnLattice4D(NC, NX, NY, NZ, NT; accelarator="cuda")
    Ug2 = MatrixOnLattice4D(NC, NX, NY, NZ, NT; accelarator="cuda")
    substitute!(Ug,U1)
    substitute!(Ug1,U2)
    substitute!(Ug2,U3)

    println(typeof(Ug))

    mul!(Ug, Ug1, Ug2)
    @time mul!(Ug, Ug1, Ug2)
    @time mul!(Ug, Ug1, Ug2)
    substitute!(U,Ug)
    val = tr(U)
    println("cuda: ",val)
    return
    val = tr(U)
    println(val)
end

test()
```

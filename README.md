# MatrixOnLattice

[![Build Status](https://github.com/cometscome/MatrixOnLattice.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cometscome/MatrixOnLattice.jl/actions/workflows/CI.yml?query=branch%3Amain)


# example

## CUDA test

```julia
using CUDA
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
    U2 = Randomfield(NC, NX, NY, NZ, NT)
    U3 = Randomfield(NC, NX, NY, NZ, NT)

    substitute!(U, U1)
    mul!(U, U2, U3)
    @time mul!(U, U2, U3)
    @time mul!(U, U2, U3)

    val = tr(U)
    println("cpu: ", val)


    Ug = MatrixOnLattice4D(NC, NX, NY, NZ, NT; accelarator="cuda")
    Ug1 = MatrixOnLattice4D(NC, NX, NY, NZ, NT; accelarator="cuda")
    Ug2 = MatrixOnLattice4D(NC, NX, NY, NZ, NT; accelarator="cuda")
    substitute!(Ug, U1)
    substitute!(Ug1, U2)
    substitute!(Ug2, U3)

    println(typeof(Ug))

    mul!(Ug, Ug1, Ug2)
    @time mul!(Ug, Ug1, Ug2)
    @time mul!(Ug, Ug1, Ug2)
    substitute!(U, Ug)
    val2 = tr(U)
    println("cuda: ", val)
    @test val2 ≈ val

end
test()
```

## JACC test
[JACC](https://github.com/JuliaORNL/JACC.jl) version

```
CPU/GPU performance portable layer for Julia

JACC.jl follows a function as a argument approach in combination with the power of Julia's ecosystem for multiple dispatch, GPU access via JuliaGPU back ends, and package extensions since Julia v1.9 . Similar to portable layers like Kokkos, users would pass a size and a function including its arguments to a parallel_for or parallel_reduce function. 
```

```julia
using CUDA
using JACC
JACC.@init_backend

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
    U2 = Randomfield(NC, NX, NY, NZ, NT)
    U3 = Randomfield(NC, NX, NY, NZ, NT)

    substitute!(U, U1)
    mul!(U, U2, U3)
    @time mul!(U, U2, U3)
    @time mul!(U, U2, U3)

    val = tr(U)
    println("cpu: ", val)


    Ug = MatrixOnLattice4D(NC, NX, NY, NZ, NT; accelarator="cuda")
    Ug1 = MatrixOnLattice4D(NC, NX, NY, NZ, NT; accelarator="cuda")
    Ug2 = MatrixOnLattice4D(NC, NX, NY, NZ, NT; accelarator="cuda")
    substitute!(Ug, U1)
    substitute!(Ug1, U2)
    substitute!(Ug2, U3)

    println(typeof(Ug))

    mul!(Ug, Ug1, Ug2)
    @time mul!(Ug, Ug1, Ug2)
    @time mul!(Ug, Ug1, Ug2)
    substitute!(U, Ug)
    val2 = tr(U)
    println("cuda: ", val)
    @test val2 ≈ val



    Uj = MatrixOnLattice4D(NC, NX, NY, NZ, NT; accelarator="jacc")
    Uj1 = MatrixOnLattice4D(NC, NX, NY, NZ, NT; accelarator="jacc")
    Uj2 = MatrixOnLattice4D(NC, NX, NY, NZ, NT; accelarator="jacc")

    substitute!(Uj, U1)
    substitute!(Uj1, U2)
    substitute!(Uj2, U3)


    mul!(Uj, Uj1, Uj2)
    @time mul!(Uj, Uj1, Uj2)
    @time mul!(Uj, Uj1, Uj2)

    substitute!(U, Uj)

    val = tr(U)
    println("jacc: ", val)

    println(typeof(Uj))
    return
    val = tr(U)
    println(val)
end
test()
```

## Single precision
using CUDA
using MatrixOnLattice
using Test
using LinearAlgebra
```julia
function test32()
    NX = 16
    NY = 16
    NZ = 16
    NT = 16
    NC = 3

    U = MatrixOnLattice4D(NC, NX, NY, NZ, NT,dtype=ComplexF32)
    U1 = Randomfield(NC, NX, NY, NZ, NT,dtype=ComplexF32)
    U2 = Randomfield(NC, NX, NY, NZ, NT,dtype=ComplexF32)
    U3 = Randomfield(NC, NX, NY, NZ, NT,dtype=ComplexF32)

    substitute!(U, U1)
    mul!(U, U2, U3)
    @time mul!(U, U2, U3)
    @time mul!(U, U2, U3)

    val = tr(U)
    println("cpu: ", val)


    Ug = MatrixOnLattice4D(NC, NX, NY, NZ, NT; accelarator="cuda",dtype=ComplexF32)
    Ug1 = MatrixOnLattice4D(NC, NX, NY, NZ, NT; accelarator="cuda",dtype=ComplexF32)
    Ug2 = MatrixOnLattice4D(NC, NX, NY, NZ, NT; accelarator="cuda",dtype=ComplexF32)
    substitute!(Ug, U1)
    substitute!(Ug1, U2)
    substitute!(Ug2, U3)

    println(typeof(Ug))

    mul!(Ug, Ug1, Ug2)
    @time mul!(Ug, Ug1, Ug2)
    @time mul!(Ug, Ug1, Ug2)
    substitute!(U, Ug)
    val = tr(U)
    println("cuda: ", val)


end
test32()
```
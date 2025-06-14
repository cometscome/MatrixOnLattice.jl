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

@testset "MatrixOnLattice.jl" begin
    # Write your tests here.
    #test32()
    test()
end

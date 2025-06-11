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
    U2 = Identityfield(NC, NX, NY, NZ, NT)
    U3 = Identityfield(NC, NX, NY, NZ, NT)

    U = MatrixOnLattice4D(NC, NX, NY, NZ, NT; accelarator="cuda")
    println(typeof(U))

    mul!(U, U2, U3)
    val = tr(U)
    println(val)
end

@testset "MatrixOnLattice.jl" begin
    # Write your tests here.
    test()
end

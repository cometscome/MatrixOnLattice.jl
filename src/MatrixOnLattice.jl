module MatrixOnLattice
using LinearAlgebra

abstract type AbstractMatrixOnLattice end

struct MatrixOnLattice4D{NC,T} <: AbstractMatrixOnLattice
    U::T
    NC::Int64
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64

    function MatrixOnLattice4D(NC, NX, NY, NZ, NT; accelarator="none")
        Ucpu = zeros(ComplexF64, NC, NC, NX, NY, NZ, NT)
        if accelarator == "none"
            U = Ucpu
        elseif accelarator == "cuda"
            ext = Base.get_extension(@__MODULE__, :CUDAExt)
            if !isnothing(ext)
                if ext.CUDA.has_cuda()
                    U = ext.CUDA.CuArray(Ucpu)
                else
                    U = Ucpu
                    @warn("CUDA is not available, using CPU array instead.")
                end
            else
                error("CUDA should be installed to use CUDAExt")
            end
        else

            error("Unsupported accelerator: $accelarator")
        end
        T = typeof(U)
        return new{NC,T}(U, NC, NX, NY, NZ, NT)
    end
end

export MatrixOnLattice4D

function Base.zero(M::MatrixOnLattice4D)
    return MatrixOnLattice4D(M.NC, M.NX, M.NY, M.NZ, M.NT)
end

function Base.similar(M::MatrixOnLattice4D)
    return MatrixOnLattice4D(M.NC, M.NX, M.NY, M.NZ, M.NT)
end

function applyfunction!(M::MatrixOnLattice4D, f!::Function)
    error("applyfunction!(M,f!) is not implemented for the type $(typeof(M)).")
end

function applyfunction!(M::MatrixOnLattice4D{NC,Array{T,6}},
    f!::Function) where {NC,T}
    for it = 1:M.NT
        for iz = 1:M.NZ
            for iy = 1:M.NY
                for ix = 1:M.NX
                    f!(M, ix, iy, iz, it)
                end
            end
        end
    end
    return
end

function applyfunction!(M::MatrixOnLattice4D,
    A::MatrixOnLattice4D, f!::Function)
    error("applyfunction!(M,A,f!) is not implemented for the type $(typeof(M)).")
end

function applyfunction!(M::MatrixOnLattice4D{NC,Array{T,6}},
    A::MatrixOnLattice4D, f!::Function) where {NC,T}
    for it = 1:M.NT
        for iz = 1:M.NZ
            for iy = 1:M.NY
                for ix = 1:M.NX
                    f!(M, A, ix, iy, iz, it)
                end
            end
        end
    end
    return
end

function applyfunction!(M::MatrixOnLattice4D,
    A::MatrixOnLattice4D, B::MatrixOnLattice4D, f!::Function)
    error("applyfunction!(M,A,B,f!) is not implemented for the type $(typeof(M)).")
end

function applyfunction!(M::MatrixOnLattice4D{NC,Array{T,6}},
    A::MatrixOnLattice4D, B::MatrixOnLattice4D, f!::Function) where {NC,T}
    for it = 1:M.NT
        for iz = 1:M.NZ
            for iy = 1:M.NY
                for ix = 1:M.NX
                    f!(M, A, B, ix, iy, iz, it)
                end
            end
        end
    end
    return
end

function applyfunctionsum(M::MatrixOnLattice4D, f::Function)
    error("applyfunction!(M,f!) is not implemented for the type $(typeof(M)).")
end

function applyfunctionsum(M::MatrixOnLattice4D{NC,Array{T,6}}, f::Function) where {NC,T}
    val = 0.0im
    for it = 1:M.NT
        for iz = 1:M.NZ
            for iy = 1:M.NY
                for ix = 1:M.NX
                    val += f(M, ix, iy, iz, it)
                end
            end
        end
    end
    return val
end

function randomize!(M::MatrixOnLattice4D{NC,T}, ix, iy, iz, it) where {NC,T}
    for j in 1:NC
        for i in 1:NC
            M.U[i, j, ix, iy, iz, it] = rand(ComplexF64)
        end
    end
end

function Randomfield(NC, NX, NY, NZ, NT)
    M = MatrixOnLattice4D(NC, NX, NY, NZ, NT)
    applyfunction!(M, randomize!)
    return M
end

function identity!(M::MatrixOnLattice4D{NC,T}, ix, iy, iz, it) where {NC,T}
    for j in 1:NC
        for i in 1:NC
            M.U[i, j, ix, iy, iz, it] = ifelse(i == j, 1.0, 0.0)
        end
    end
end

function multiply!(C::MatrixOnLattice4D{NC,T},
    A::MatrixOnLattice4D, B::MatrixOnLattice4D, ix, iy, iz, it) where {NC,T}
    for j in 1:NC
        for i in 1:NC
            C.U[i, j, ix, iy, iz, it] = 0.0
            for k in 1:NC
                C.U[i, j, ix, iy, iz, it] += A.U[i, k, ix, iy, iz, it] * B.U[k, j, ix, iy, iz, it]
            end
        end
    end
end

function Identityfield(NC, NX, NY, NZ, NT)
    M = MatrixOnLattice4D(NC, NX, NY, NZ, NT)
    applyfunction!(M, identity!)
    return M
end
export Identityfield, Randomfield

function LinearAlgebra.mul!(C::MatrixOnLattice4D, A::MatrixOnLattice4D, B::MatrixOnLattice4D)
    applyfunction!(C, A, B, multiply!)
    return C
end

function traceonlattice(M::MatrixOnLattice4D{NC,T}, ix, iy, iz, it) where {NC,T}
    val = 0.0im
    for i in 1:NC
        val += M.U[i, i, ix, iy, iz, it]
    end
    return val
end

function LinearAlgebra.tr(M::MatrixOnLattice4D)
    val = applyfunctionsum(M, traceonlattice)
    return val
end

end
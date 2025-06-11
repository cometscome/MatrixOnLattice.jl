module MatrixOnLattice
using LinearAlgebra

include("Blockindex.jl")

abstract type AbstractMatrixOnLattice end

struct MatrixOnLattice4D{NC,T,accdevise,dtype} <: AbstractMatrixOnLattice
    U::T
    NC::Int64
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    blockinfo::Union{Blockindices,Nothing}
end

export MatrixOnLattice4D
export Identityfield, Randomfield, substitute!

function MatrixOnLattice4D(NC, NX, NY, NZ, NT;
    accelarator="none",
    blocks_in=nothing, dtype=ComplexF64)

    if accelarator == "none"
        return MatrixOnLattice4D_cpu(NC, NX, NY, NZ, NT; dtype)
    elseif accelarator == "cuda"
        return MatrixOnLattice4D_cuda(NC, NX, NY, NZ, NT, blocks_in; dtype)
    elseif accelarator == "jacc"
        return MatrixOnLattice4D_jacc(NC, NX, NY, NZ, NT; dtype)
    end

end

function MatrixOnLattice4D_cpu(NC, NX, NY, NZ, NT; dtype=ComplexF64)
    Ucpu = zeros(dtype, NC, NC, NX, NY, NZ, NT)
    T = typeof(Ucpu)
    accdevise = :none
    return MatrixOnLattice4D{NC,T,accdevise,dtype}(Ucpu, NC, NX, NY, NZ, NT, nothing)
end

function MatrixOnLattice4D_cuda(NC, NX, NY, NZ, NT, blocks_in; dtype=ComplexF64)

    ext = Base.get_extension(@__MODULE__, :CUDAExt)
    if !isnothing(ext)
        if ext.CUDA.has_cuda()
            L = (NX, NY, NZ, NT)
            if blocks_in !== nothing
                blocks = blocks_in
            else
                blocks = (4, 4, 4, 4)
            end

            blockinfo = Blockindices(L, blocks)#Blockindices(Tuple(blocks),Tuple(blocks_s),Tuple(blocknumbers),Tuple(blocknumbers_s),blocksize,rsize)
            blocksize = blockinfo.blocksize
            rsize = blockinfo.rsize
            Ucpu = zeros(dtype, NC, NC, blocksize, rsize)
            accdevise = :cuda
            U = ext.CUDA.CuArray(Ucpu)
        else
            U = zeros(dtype, NC, NC, NX, NY, NZ, NT)
            accdevise = :none
            blockinfo = nothing
            @warn("CUDA is not available, using CPU array instead.")
        end
    else
        error("CUDA should be installed to use CUDAExt")
    end

    T = typeof(U)
    return MatrixOnLattice4D{NC,T,accdevise,dtype}(U, NC, NX, NY, NZ, NT, blockinfo)
end

function MatrixOnLattice4D_jacc(NC, NX, NY, NZ, NT; dtype=ComplexF64)
    ext = Base.get_extension(@__MODULE__, :JACCExt)
    if !isnothing(ext)
        NV = NX * NY * NZ * NT
        Ucpu = zeros(dtype, NC, NC, NV)
        U = ext.JACC.array(Ucpu)
        accdevise = :jacc
    else
        error("JACC should be installed to use JACCExt")
    end
    T = typeof(U)
    return MatrixOnLattice4D{NC,T,accdevise,dtype}(U, NC, NX, NY, NZ, NT, nothing)
end




function Base.zero(M::MatrixOnLattice4D{NC,T,:none,dtype}) where {NC,T,dtype}
    return MatrixOnLattice4D_cpu(M.NC, M.NX, M.NY, M.NZ, M.NT; dtype)
end

function Base.zero(M::MatrixOnLattice4D{NC,T,:cuda,dtype}) where {NC,T,dtype}
    return MatrixOnLattice4D_cuda(M.NC, M.NX, M.NY, M.NZ, M.NT, M.blockinfo; dtype)
end

function Base.zero(M::MatrixOnLattice4D{NC,T,:jacc,dtype}) where {NC,T,dtype}
    return MatrixOnLattice4D_jacc(M.NC, M.NX, M.NY, M.NZ, M.NT; dtype)
end


function Base.similar(M::MatrixOnLattice4D{NC,T,accdevise}) where {NC,T,accdevise}
    return zero(M)
end

function applyfunction!(M::MatrixOnLattice4D, f!::Function)
    error("applyfunction!(M,f!) is not implemented for the type $(typeof(M)).")
end

function applyfunction!(M::MatrixOnLattice4D{NC,Array{T,6},:none},
    f!::Function) where {NC,T}
    for it = 1:M.NT
        for iz = 1:M.NZ
            for iy = 1:M.NY
                for ix = 1:M.NX
                    f!(ix, iy, iz, it, M)
                end
            end
        end
    end
    return
end

function applyfunction!(M::MatrixOnLattice4D,
    A::MatrixOnLattice4D, f!::Function)
    error("applyfunction!(M,A,f!) is not implemented for the type $(typeof(M)) and $(typeof(A)).")
end

function applyfunction!(M::MatrixOnLattice4D{NC,Array{T,6},:none},
    A::MatrixOnLattice4D{NC,Array{T,6},:none}, f!::Function) where {NC,T}
    for it = 1:M.NT
        for iz = 1:M.NZ
            for iy = 1:M.NY
                for ix = 1:M.NX
                    f!(ix, iy, iz, it, M, A)
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

function applyfunction!(M::MatrixOnLattice4D{NC,Array{T,6},:none},
    A::MatrixOnLattice4D, B::MatrixOnLattice4D, f!::Function) where {NC,T}
    for it = 1:M.NT
        for iz = 1:M.NZ
            for iy = 1:M.NY
                for ix = 1:M.NX
                    f!(ix, iy, iz, it, M, A, B)
                end
            end
        end
    end
    return
end

function applyfunctionsum(M::MatrixOnLattice4D, f::Function)
    error("applyfunction!(M,f!) is not implemented for the type $(typeof(M)).")
end

function applyfunctionsum(M::MatrixOnLattice4D{NC,Array{T,6},:none}, f::Function) where {NC,T}
    val = 0.0im
    for it = 1:M.NT
        for iz = 1:M.NZ
            for iy = 1:M.NY
                for ix = 1:M.NX
                    val += f(ix, iy, iz, it, M)
                end
            end
        end
    end
    return val
end

function randomize!(ix::TN, iy::TN, iz::TN, it::TN, M::MatrixOnLattice4D{NC,T}) where {NC,T,TN<:Integer}
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

function identity!(ix::TN, iy::TN, iz::TN, it::TN, M::MatrixOnLattice4D{NC,T}) where {NC,T,TN<:Integer}
    for j in 1:NC
        for i in 1:NC
            M.U[i, j, ix, iy, iz, it] = ifelse(i == j, 1.0, 0.0)
        end
    end
end

function multiply!(ix::TN, iy::TN, iz::TN, it::TN, C::MatrixOnLattice4D{NC,T},
    A::MatrixOnLattice4D, B::MatrixOnLattice4D) where {NC,T,TN<:Integer}
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


function LinearAlgebra.mul!(C::MatrixOnLattice4D, A::MatrixOnLattice4D, B::MatrixOnLattice4D)
    applyfunction!(C, A, B, multiply!)
    return C
end

function traceonlattice(ix::TN, iy::TN, iz::TN, it::TN, M::MatrixOnLattice4D{NC,T}) where {NC,T,TN<:Integer}
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

function substitute_each!(ix::TN, iy::TN, iz::TN, it::TN, M::MatrixOnLattice4D{NC,T},
    A::MatrixOnLattice4D) where {NC,T,TN<:Integer}
    for j in 1:NC
        for i in 1:NC
            M.U[i, j, ix, iy, iz, it] = A.U[i, j, ix, iy, iz, it]
        end
    end
end

function substitute!(M::MatrixOnLattice4D, A::MatrixOnLattice4D)
    applyfunction!(M, A, substitute_each!)
end

end
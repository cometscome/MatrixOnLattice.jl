module CUDAExt
using MatrixOnLattice
import CUDA
import JACC
using LinearAlgebra



function MatrixOnLattice.applyfunction!(M::MatrixOnLattice4D{NC,T,:cuda},
    A::MatrixOnLattice4D, B::MatrixOnLattice4D, f!::Function) where {NC,T}
    CUDA.@sync begin
        CUDA.@cuda threads = M.blockinfo.blocksize blocks = M.blockinfo.rsize f!(M.blockinfo, M.U, A.U, B.U,NC)
    end
    return
end


function MatrixOnLattice.applyfunction!(M::MatrixOnLattice4D{3,T,:cuda},
    A::MatrixOnLattice4D, B::MatrixOnLattice4D, f!::Function) where {T}
    CUDA.@sync begin
        CUDA.@cuda threads = M.blockinfo.blocksize blocks = M.blockinfo.rsize f!(M.blockinfo, M.U, A.U, B.U)
    end
    return
end


function MatrixOnLattice.applyfunction!(M::MatrixOnLattice4D{NC,T1,:cuda},
    A::MatrixOnLattice4D{NC,T2,:none}, f!::Function) where {NC,T1,T2}

    Mcpu = Array(M.U)
    blockinfo = M.blockinfo
    for r = 1:blockinfo.rsize
        for b = 1:blockinfo.blocksize
            ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)
             f!(b,r,ix, iy, iz, it, Mcpu, A)
        end
    end
    Mgpu = CUDA.CuArray(Mcpu)
    M.U .= Mgpu
    return
end

function MatrixOnLattice.applyfunction!(M::MatrixOnLattice4D{NC,T1,:none},
    A::MatrixOnLattice4D{NC,T2,:cuda}, f!::Function) where {NC,T1,T2}

    Acpu = Array(A.U)
    blockinfo = A.blockinfo
    for r = 1:blockinfo.rsize
        for b = 1:blockinfo.blocksize
            ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)
             f!(b,r,ix, iy, iz, it, M, Acpu)
        end
    end
    return
end

function MatrixOnLattice.substitute_each!(b::TN,r::TN,ix::TN, iy::TN, iz::TN, it::TN, M::Array{T,4},
    A::MatrixOnLattice4D{NC,T2}) where {NC,T,TN<:Integer,T2}
    for j in 1:NC
        for i in 1:NC
            M[i, j,b,r] = A.U[i, j, ix, iy, iz, it]
        end
    end
end

function MatrixOnLattice.substitute_each!(b::TN,r::TN,ix::TN, 
    iy::TN, iz::TN, it::TN, M::MatrixOnLattice4D{NC,T2},
    A::Array{T,4}) where {NC,T,TN<:Integer,T2}
    for j in 1:NC
        for i in 1:NC
            M.U[i, j, ix, iy, iz, it] = A[i, j,b,r]
        end
    end
end


function MatrixOnLattice.multiply!(block::Blockindices, C::T1,
    A::T1, B::T1,NC) where {T1}
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    for j in 1:NC
        for i in 1:NC
            C[i, j, b, r] = 0.0
            for k in 1:NC
                C[i, j, b, r] += A[i, k, b, r] * B[k, j, b, r]
            end
        end
    end
end


function LinearAlgebra.mul!(C::MatrixOnLattice4D{3,T,:cuda},
     A::MatrixOnLattice4D, B::MatrixOnLattice4D) where {T}
    MatrixOnLattice.applyfunction!(C, A, B, MatrixOnLattice.multiply3!)
    return C
end

function MatrixOnLattice.multiply3!(block::Blockindices, C::T1,
    A::T1, B::T1) where {T1}
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    C[1, 1, b, r] = A[1, 1, b, r] * B[1, 1, b, r] + 
             A[1, 2, b, r] * B[2, 1, b, r] +
              A[1, 3, b, r] * B[3, 1, b, r]
    C[2, 1, b, r] = A[2, 1, b, r] * B[1, 1, b, r] +
             A[2, 2, b, r] * B[2, 1, b, r] +
              A[2, 3, b, r] * B[1, 3, b, r]
    C[3, 1, b, r] = A[3, 1, b, r] * B[1, 1, b, r] +
             A[3, 2, b, r] * B[2, 1, b, r] +
              A[3, 3, b, r] * B[1, 3, b, r]              

    C[1, 2, b, r] = A[1, 1, b, r] * B[1, 2, b, r] +
             A[1, 2, b, r] * B[2, 2, b, r] +
              A[1, 3, b, r] * B[3, 2, b, r]
    C[2, 2, b, r] = A[2, 1, b, r] * B[1, 2, b, r] +
             A[2, 2, b, r] * B[2, 2, b, r] +
              A[2, 3, b, r] * B[3, 2, b, r]
    C[3, 2, b, r] = A[3, 1, b, r] * B[1, 2, b, r] +
             A[3, 2, b, r] * B[2, 2, b, r] +
              A[3, 3, b, r] * B[3, 2, b, r]

              
    C[1, 3, b, r] = A[1, 1, b, r] * B[1, 3, b, r] +
             A[1, 2, b, r] * B[2, 3, b, r] +
              A[1, 3, b, r] * B[3, 3, b, r]


    C[2, 3, b, r] = A[2, 1, b, r] * B[1, 3, b, r] +
             A[2, 2, b, r] * B[2, 3, b, r] +
              A[2, 3, b, r] * B[3, 3, b, r]


    C[3, 3, b, r] = A[3, 1, b, r] * B[1, 3, b, r] +
             A[3, 2, b, r] * B[2, 3, b, r] +
              A[3, 3, b, r] * B[3, 3, b, r]

    #=
    C[i, j, b, r] += A[i, k, b, r] * B[k, j, b, r]
    C[i, j, b, r] += A[i, k, b, r] * B[k, j, b, r]
    C[i, j, b, r] += A[i, k, b, r] * B[k, j, b, r]
    C[i, j, b, r] += A[i, k, b, r] * B[k, j, b, r]
    C[i, j, b, r] += A[i, k, b, r] * B[k, j, b, r]
    C[i, j, b, r] += A[i, k, b, r] * B[k, j, b, r]
    for j in 1:3
        for i in 1:3
            C[i, j, b, r] = 0.0
            for k in 1:3
                C[i, j, b, r] += A[i, k, b, r] * B[k, j, b, r]
            end
        end
    end
    =#
    return
end




end
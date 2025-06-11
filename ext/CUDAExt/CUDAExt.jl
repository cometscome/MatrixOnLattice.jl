module CUDAExt
using MatrixOnLattice
import CUDA

function MatrixOnLattice.applyfunction!(M::MatrixOnLattice4D{NC,T,:cuda},
    A::MatrixOnLattice4D, B::MatrixOnLattice4D, f!::Function) where {NC,T}
    CUDA.@sync begin
        CUDA.@cuda threads = M.blockinfo.blocksize blocks = M.blockinfo.rsize f!(M.blockinfo, M, A, B)
    end
    return
end

function multiply!(block::Blockindices, C::MatrixOnLattice4D{NC,T,:cuda},
    A::MatrixOnLattice4D, B::MatrixOnLattice4D) where {NC,T}
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    for j in 1:NC
        for i in 1:NC
            C.U[i, j, b, r] = 0.0
            for k in 1:NC
                C.U[i, j, b, r] += A.U[i, k, b, r] * B.U[k, j, b, r]
            end
        end
    end
end


end
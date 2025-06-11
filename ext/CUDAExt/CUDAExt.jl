module CUDAExt
using MatrixOnLattice
import CUDA

function MatrixOnLattice.applyfunction!(M::MatrixOnLattice4D{NC,T},
    A::MatrixOnLattice4D, B::MatrixOnLattice4D, f!::Function) where {NC,T <: CUDA.CuArray}
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
end
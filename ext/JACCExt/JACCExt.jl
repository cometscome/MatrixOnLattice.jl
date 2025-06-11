module JACCExt
using MatrixOnLattice
import JACC

const JACCArray = JACC.array_type()

function MatrixOnLattice.applyfunction!(M::MatrixOnLattice4D{NC,T,:jacc},
    A::MatrixOnLattice4D, B::MatrixOnLattice4D, f!::Function) where {NC,T}
    N = M.NX * M.NY * M.NZ * M.NT

    JACC.parallel_for(N, f!, M.U, A.U, B.U, NC)
    #CUDA.@sync begin
    #    CUDA.@cuda threads = M.blockinfo.blocksize blocks = M.blockinfo.rsize f!(M.blockinfo, M.U, A.U, B.U,NC)
    #end
    return
end

function MatrixOnLattice.applyfunction!(M::MatrixOnLattice4D{NC,T1,:jacc},
    A::MatrixOnLattice4D{NC,T2,:none}, f!::Function) where {NC,T1,T2}

    Mcpu = Array(M.U)
    #=
    blockinfo = M.blockinfo
    for r = 1:blockinfo.rsize
        for b = 1:blockinfo.blocksize
            ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)
             f!(b,r,ix, iy, iz, it, Mcpu, A)
        end
    end
    =#
    Mgpu = JACC.array(Mcpu)
    M.U .= Mgpu
    return
end

function MatrixOnLattice.applyfunction!(M::MatrixOnLattice4D{NC,T1,:none},
    A::MatrixOnLattice4D{NC,T2,:jacc}, f!::Function) where {NC,T1,T2}

    Acpu = Array(A.U)
    blockinfo = A.blockinfo
    for r = 1:blockinfo.rsize
        for b = 1:blockinfo.blocksize
            ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)
            f!(b, r, ix, iy, iz, it, M, Acpu)
        end
    end
    return
end

end
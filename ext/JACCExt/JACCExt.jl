module JACCExt
using MatrixOnLattice
import JACC

const JACCArray = JACC.array_type()

function index_to_coords(i, NX, NY, NZ, NT)
    ix = mod1(i, NX)
    iy = mod1(div(i - 1, NX) + 1, NY)
    iz = mod1(div(i - 1, NX * NY) + 1, NZ)
    it = div(i - 1, NX * NY * NZ) + 1
    return ix, iy, iz, it
end

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
    N = M.NX * M.NY * M.NZ * M.NT

    Mcpu = Array(M.U)
    for i = 1:N
        ix, iy, iz, it = index_to_coords(i, M.NX, M.NY, M.NZ, M.NT)
        f!(i, ix, iy, iz, it, Mcpu, A)
    end
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
    N = M.NX * M.NY * M.NZ * M.NT
    for i = 1:N
        ix, iy, iz, it = index_to_coords(i, M.NX, M.NY, M.NZ, M.NT)
        f!(i, ix, iy, iz, it, M, Acpu)
    end
    #=
    blockinfo = A.blockinfo
    for r = 1:blockinfo.rsize
        for b = 1:blockinfo.blocksize
            ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)
            f!(b, r, ix, iy, iz, it, M, Acpu)
        end
    end
    =#
    return
end

function MatrixOnLattice.substitute_each!(i::TN, ix::TN, iy::TN, iz::TN, it::TN, M::Array{T,3},
    A::MatrixOnLattice4D{NC,T2}) where {NC,T,TN<:Integer,T2}
    for jc in 1:NC
        for ic in 1:NC
            M[ic, jc, i] = A.U[ic, jc, ix, iy, iz, it]
        end
    end
end


function MatrixOnLattice.substitute_each!(i::TN, ix::TN,
    iy::TN, iz::TN, it::TN, M::MatrixOnLattice4D{NC,T2},
    A::Array{T,3}) where {NC,T,TN<:Integer,T2}
    for jc in 1:NC
        for ic in 1:NC
            M.U[ic, jc, ix, iy, iz, it] = A[ic, jc, i]
        end
    end
end


function MatrixOnLattice.multiply!(i::Integer, C::T1,
    A::T1, B::T1, NC) where {T1}
    for jc in 1:NC
        for ic in 1:NC
            C[ic, jc, i] = 0.0
            for kc in 1:NC
                C[ic, jc, i] += A[ic, kc, i] * B[kc, jc, i]
            end
        end
    end
end


end
struct Blockindices
    blocks::NTuple{4,Int64}
    blocks_s::NTuple{4,Int64}
    blocknumbers::NTuple{4,Int64}
    blocknumbers_s::NTuple{4,Int64}
    blocksize::Int64 #num. of Threads 
    rsize::Int64 #num. of blocks

    function Blockindices(L, blocks)
        blocknumbers = div.(L, blocks)

        dim = length(L)
        blocks_s = ones(dim)
        blocknumbers_s = ones(dim)
        for i in 2:dim
            for j in 1:i-1
                blocknumbers_s[i] = blocknumbers_s[i] * blocknumbers[j]
                blocks_s[i] = blocks_s[i] * blocks[j]
            end
        end

        blocksize = prod(blocks)
        rsize = prod(blocknumbers)

        return new(Tuple(blocks), Tuple(blocks_s), Tuple(blocknumbers), Tuple(blocknumbers_s), blocksize, rsize)

    end
end
export Blockindices

@inline convert_x(x, xd, xd_s) = mod(div(x - 1, xd_s), xd)
@inline convert_br(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s) = 1 + convert_x(b, blocks, blocks_s) +
                                                                           convert_x(r, blocknumbers, blocknumbers_s) * blocks

function fourdim_cordinate(b, r, blockinfo)
    blocks = blockinfo.blocks[1]
    blocks_s = blockinfo.blocks_s[1]
    blocknumbers = blockinfo.blocknumbers[1]
    blocknumbers_s = blockinfo.blocknumbers_s[1]
    ix = convert_br(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)

    blocks = blockinfo.blocks[2]
    blocks_s = blockinfo.blocks_s[2]
    blocknumbers = blockinfo.blocknumbers[2]
    blocknumbers_s = blockinfo.blocknumbers_s[2]
    iy = convert_br(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)

    blocks = blockinfo.blocks[3]
    blocks_s = blockinfo.blocks_s[3]
    blocknumbers = blockinfo.blocknumbers[3]
    blocknumbers_s = blockinfo.blocknumbers_s[3]
    iz = convert_br(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)

    blocks = blockinfo.blocks[4]
    blocks_s = blockinfo.blocks_s[4]
    blocknumbers = blockinfo.blocknumbers[4]
    blocknumbers_s = blockinfo.blocknumbers_s[4]
    it = convert_br(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)

    return ix, iy, iz, it
end
export  fourdim_cordinate

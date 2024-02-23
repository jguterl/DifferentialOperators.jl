export IndexIterator

#
# These are all index related structs created by Jerome
#

#
# Allow 32 bit integers (likely needed for GPU)
#
Index = Union{Int64,Int32}

#
# Needs to be another module / file entirely
#
struct IndexIterator{I}
    start :: I
    stop  :: I
end

#
# This is a very stupid name, almost identical to the other one
# 
struct IndexIterators{I,J,K}
    i :: I
    j :: J
    k :: K
end

#
# Likely not needed or another module / file
#
struct GhostCells
    nx :: Int64
    ny :: Int64
    nz :: Int64
end

#
# What is this for? Likely another module / file entirely
#
struct GridIndexes{I<:IndexIterators}
    gc         :: GhostCells
    inner_iter :: I
    outer_iter :: I
end

function GridIndexes(dims::NTuple{N,Int64}; nx_gc=1, ny_gc=1, nz_gc=1, kw...) where N
    nx, ny, nz = get_dims(dims)
    gc = GhostCells(nx_gc, ny_gc, nz_gc)
    inner_iter = IndexIterators(IndexIterator(1, nx, nx_gc), IndexIterator(1, ny, ny_gc), IndexIterator(1, nz, nz_gc))
    outer_iter = IndexIterators(IndexIterator(1, nx, 0), IndexIterator(1, ny, 0), IndexIterator(1,nz, 0))
    GridIndexes(gc, inner_iter, outer_iter)
end

IndexIterator(r::UnitRange) = IndexIterator(r.start,r.stop)
IndexIterator(n_start::Int64, n::Int64  , n_gc::Int64) = n_start+n_gc:n-n_gc
IndexIterator(n_start::Int64, n::Missing, n_gc::Int64) = missing

get_dims(dims::NTuple{1,Int64}) = dims[1], missing, missing
get_dims(dims::NTuple{2,Int64}) = dims[1], dims[2], missing
get_dims(dims::NTuple{3,Int64}) = dims[1], dims[2], dims[3]

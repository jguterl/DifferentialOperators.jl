const Grid1D{X}     = Grid{X,Missing,Missing}
const Grid2D{X,Y}   = Grid{X,Y,Missing}
const Grid3D{X,Y,Z} = Grid{X,Y,Z}


#
# Needs to be a different module / file entirely
#
struct NormalVectors{SS,NN,WW,EE}
    S :: SS
    N :: NN
    W :: WW
    E :: EE
end 

function NormalVectors(grid::Grid) 
    N = VectorField(ones(grid), zeros(grid), zeros(grid))
    S = VectorField(-ones(grid), zeros(grid), zeros(grid))
    W = VectorField(zeros(grid), -ones(grid), zeros(grid))
    E = VectorField(zeros(grid), ones(grid), zeros(grid))
    return NormalVectors(N,S,W,E)
end

UnitaryVectors(grid::Grid) = VectorField(VectorField(ones(grid), zeros(grid),zeros(grid)),VectorField(zeros(grid), ones(grid),zeros(grid)),VectorField(zeros(grid), zeros(grid),ones(grid)))
export MHDGrid


#
# All this needs to be a different struct / file entirely
#
struct MHDGrid{G,D,I,N,U} <: AbstractMHDGrid
    grid::G
    grid_data::D
    indexes::I
    n::N #normal vector
    e::U #unitary vectors
end
MHDGrid(nx::Int64, ny::Int64; kw...) = MHDGrid((nx,ny); kw...)
function MHDGrid(dims::NTuple{N,Int64}; kw...) where N
    grid = Grid(dims; kw...) 
    indexes = GridIndexes(dims; kw...)
    grid_data = GridDerivatives(grid, indexes.ghost_cells)
    n = NormalVectors(grid)
    e = UnitaryVectors(grid)
    MHDGrid(grid, grid_data, indexes, n, e)
end


# TODO: FH check that
#function set_dx_ghost_cells!(dx, ghost_cells::GhostCells) 
#    ghost_cells.nx == 0 && return 
#    dx[1:1+ghost_cells.nx-1,:]  = dx[1+ghost_cells.nx,:] 
#    dx[end-ghost_cells.nx+1:end,:]  = dx[end-ghost_cells.nx,:]
#end

# TODO: FH check that
#function set_dy_ghost_cells!(dy, ghost_cells::GhostCells)
#    ghost_cells.ny == 0 && return
#    dy[:,1:1+ghost_cells.ny-1] = dy[:,1+ghost_cells.ny]
#    dy[:,end-ghost_cells.ny+1:end] = dy[:,end-ghost_cells.ny]
#end

#
# Needs to be another module / file entirely
#
struct IndexIterator{I}
    start:: I
    stop :: I
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
    nx::Int64
    ny::Int64
    nz::Int64
end

#
# What is this for? Likely another module / file entirely
#
struct GridIndexes{I<:IndexIterators}
    gc :: GhostCells
    inner_iter::I
    outer_iter::I
end

function GridIndexes(dims::NTuple{N,Int64}; nx_gc=1, ny_gc=1, nz_gc=1, kw...) where N
    nx, ny, nz = get_dims(dims)
    gc = GhostCells(nx_gc, ny_gc, nz_gc)
    inner_iter = IndexIterators(IndexIterator(1,nx, nx_gc), IndexIterator(1,ny, ny_gc), IndexIterator(1,nz, nz_gc))
    outer_iter = IndexIterators(IndexIterator(1,nx, 0), IndexIterator(1,ny, 0), IndexIterator(1,nz, 0))
    GridIndexes(gc, inner_iter, outer_iter)
end

# mhd grid ----



IndexIterator(r::UnitRange) = IndexIterator(r.start,r.stop)
IndexIterator(n_start::Int64, n::Int64  , n_gc::Int64) = n_start+n_gc:n-n_gc
IndexIterator(n_start::Int64, n::Missing, n_gc::Int64) = missing

get_dims(dims::NTuple{1,Int64}) = dims[1], missing, missing
get_dims(dims::NTuple{2,Int64}) = dims[1], dims[2], missing
get_dims(dims::NTuple{3,Int64}) = dims[1], dims[2], dims[3]
export IndexIterator

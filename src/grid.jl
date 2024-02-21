# a generic uniform grid generator just for testing ... 
Base.@kwdef struct Grid{X,Y,Z} <: AbstractGrid
    x::X = missing
    y::Y = missing
    z::Z = missing
end
Base.ones(grid::Grid) = ones(grid,current_backend.value)
Base.zeros(grid::Grid) = zeros(grid, current_backend.value)
Base.ones(grid::Grid, backend::Backend) = backend(ones(size(grid.x)...))
Base.zeros(grid::Grid, backend::Backend) = backend(zeros(size(grid.x)...))
const Grid1D{X} = Grid{X,Missing,Missing}
const Grid2D{X,Y} = Grid{X,Y,Missing}
const Grid3D{X,Y,Z} = Grid{X,Y,Z}

_get_grid_points(i::Int64, dims, ng, d0, L) = (d0[i] + L[i]) / (dims[i] - 1) * (getindex.(collect(Iterators.product((1-ng[i]:d+ng[i] for d in dims)...)), i) .- 1)
get_grid_points(backend::CPUBackend, args..., ) = _get_grid_points(args...)
get_grid_points(backend::CUDABackend, args..., ) = CUDA.CuArray(_get_grid_points(args...))
Grid(dims::NTuple{N,Int64}; L=[1.0, 1.0, 1.0], ng=[0, 0, 0], d0=[0.0, 0.0, 0.0], backend::Backend=current_backend.value) where {N} = Grid(; (fn => get_grid_points(backend, i, dims, ng, d0, L) for ((i, d), fn) in zip(enumerate(dims), fieldnames(Grid)))...)
Grid(nx::Int64, ny::Int64; kw...) = Grid((nx,ny); kw...)
Base.size(grid::Grid) = size(grid.x)

# mhd grid ----
struct IndexIterators{I,J,K}
    i :: I
    j :: J
    k :: K
end

struct GhostCells
    nx::Int64
    ny::Int64
    nz::Int64
end
struct GridIndexes{I<:IndexIterators}
    gc :: GhostCells
    inner_iter::I
    outer_iter::I
end

struct IndexIterator{I}
    start:: I
    stop :: I
end
IndexIterator(r::UnitRange) = IndexIterator(r.start,r.stop)
get_dims(dims::NTuple{1,Int64}) = dims[1], missing, missing
get_dims(dims::NTuple{2,Int64}) = dims[1], dims[2], missing
get_dims(dims::NTuple{3,Int64}) = dims[1], dims[2], dims[3]
IndexIterator(n_start::Int64, n::Int64, n_gc::Int64) = n_start+n_gc:n-n_gc
IndexIterator(n_start::Int64, n::Missing, n_gc::Int64) = missing
export IndexIterator

function GridIndexes(dims::NTuple{N,Int64}; nx_gc=1, ny_gc=1, nz_gc=1, kw...) where N
    nx,ny, nz = get_dims(dims)
    gc = GhostCells(nx_gc, ny_gc, nz_gc)
    inner_iter = IndexIterators(IndexIterator(1,nx, nx_gc), IndexIterator(1,ny, ny_gc), IndexIterator(1,nz, nz_gc))
    outer_iter = IndexIterators(IndexIterator(1,nx, 0), IndexIterator(1,ny, 0), IndexIterator(1,nz, 0))
    GridIndexes(gc, inner_iter, outer_iter)
end






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



struct GridDerivatives{X<:GridData,Y<:GridData,Z<:GridData,V,B} <: AbstractGridDerivatives{B}
    dx :: X
    dy :: Y
    dz :: Z
    backend::B
end 
function GridDerivatives(grid::Grid, ghost_cells::GhostCells; Kx=1, Ky=1, Kz=1)
    dx = grid.x .- circshift(grid.x, (1, 0))
    dy = grid.y .- circshift(grid.y, (0, 1))
    dz = missing
    set_dx_ghost_cells!(dx, ghost_cells)
    set_dy_ghost_cells!(dy, ghost_cells)
    GridDerivatives(GridData(dx), GridData(dy), GridData(dz))
end

# TODO: FH check that
function set_dx_ghost_cells!(dx, ghost_cells::GhostCells) 
    ghost_cells.nx == 0 && return 
    dx[1:1+ghost_cells.nx-1,:]  = dx[1+ghost_cells.nx,:] 
    dx[end-ghost_cells.nx+1:end,:]  = dx[end-ghost_cells.nx,:]
end

# TODO: FH check that
function set_dy_ghost_cells!(dy, ghost_cells::GhostCells)
    ghost_cells.ny == 0 && return
    dy[:,1:1+ghost_cells.ny-1] = dy[:,1+ghost_cells.ny]
    dy[:,end-ghost_cells.ny+1:end] = dy[:,end-ghost_cells.ny]
end



GridDerivatives(x::AbstractGridData{B}, y::AbstractGridData{B}, z::AbstractGridData{B}) where {B<:Backend} = GridDerivatives(x,y ,z, B())

get_dx(grid_data::GridDerivatives) = CUDA.@allowscalar grid_data.dx.data[3, 3] #TODO: this is wrong if  ng c >2 

get_dy(grid_data::GridDerivatives) = CUDA.@allowscalar grid_data.dy.data[3, 3] # TODO: this is wrong if  ng c >2 

Adapt.@adapt_structure GridDerivatives

export get_dx, get_dy
# -- junk ----
# derivative operators
# abstract type DerivativeOperator{K} end
# struct dX{K} <: DerivativeOperator{K} end
# struct dY{K} <: DerivativeOperator{K} end
# struct dZ{K} <: DerivativeOperator{K} end

# struct ApplyDerivativeOperator{T<:DerivativeOperator,X,Y,Z}
#     dx::X
#     dy::Y
#     dz::Z
# end

# ApplyDerivativeOperator{T}(x::X, y::Y, z::Z) where {T,X,Y,Z} = ApplyDerivativeOperator{T,X,Y,Z}(x, y, z)
# function ApplyDerivativeOperator{T}(grid::Grid) where {T<:DerivativeOperator}
#     dx = grid.x .- circshift(grid.x, (1, 0))
#     dy = grid.y .- circshift(grid.y, (0, 1))
#     dz = missing
#     ApplyDerivativeOperator{T}(GridData(dx), GridData(dy), GridData(dz))
# end
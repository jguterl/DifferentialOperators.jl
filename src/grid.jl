# a generic uniform grid generator just for testing ... 
Base.@kwdef struct Grid{X,Y,Z} <: AbstractGrid
    x::X = missing
    y::Y = missing
    z::Z = missing
end
const Grid1D{X} = Grid{X,Missing,Missing}
const Grid2D{X,Y} = Grid{X,Y,Missing}
const Grid3D{X,Y,Z} = Grid{X,Y,Z}

Grid(dims::NTuple{N,Int64}; L=[1.0, 1.0, 1.0], d0=[0.0, 0.0, 0.0]) where {N} = Grid(; (fn => (d0[i] + L[i]) / (dims[i] - 1) * (getindex.(collect(Iterators.product((1:d for d in dims)...)), i) .- 1) for ((i, d), fn) in zip(enumerate(dims), fieldnames(Grid)))...)
Grid(nx::Int64, ny::Int64; kw...) = Grid((nx,ny); kw...)

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

get_dims(dims::NTuple{1,Int64}) = dims[1], missing, missing
get_dims(dims::NTuple{2,Int64}) = dims[1], dims[2], missing
get_dims(dims::NTuple{3,Int64}) = dims[1], dims[2], dims[3]
IndexIterator(n::Int64, n_gc::Int64) = 1+n_gc:n-n_gc
function GridIndexes(dims::NTuple{N,Int64}; nx_gc=1, ny_gc=1, nz_gc=1) where N
    nx,ny, nz = get_dims(dims...)
    gc = GhostCells(nx_gc, ny_gc, nz_gc)
    inner_iter = IndexIterators(IndexIterator(nx, nx_gc), IndexIterator(ny, ny_gc), IndexIterator(nz, nz_gc))
    outer_iter = IndexIterators(IndexIterator(nx, 0), IndexIterator(ny, 0), IndexIterator(nz, 0))
    GridIndexes(gc, inner_iter, outer_iter)
end


struct MHDGrid{G,D,I}
    grid :: G
    grid_data :: D
    indexes :: I
end

function MHDGrid(dims::NTuple{N,Int64}; kw...)
    grid = Grid(dims;kw...) 
    grid_data = GridDerivatives(grid);
    indexes = GridIndexes(dims;kw...)
    MHDGrid(grid,grid_data,indexes)
end

export MHDGrid
Base.size(grid::Grid) = size(grid.x)

# derivative operators
abstract type DerivativeOperator{K} end
struct dX{K} <: DerivativeOperator{K} end
struct dY{K} <: DerivativeOperator{K} end
struct dZ{K} <: DerivativeOperator{K} end

struct ApplyDerivativeOperator{T<:DerivativeOperator,X,Y,Z}
    dx::X
    dy::Y
    dz::Z
end

ApplyDerivativeOperator{T}(x::X, y::Y, z::Z) where {T,X,Y,Z} = ApplyDerivativeOperator{T,X,Y,Z}(x, y, z)
function ApplyDerivativeOperator{T}(grid::Grid) where {T<:DerivativeOperator}
    dx = grid.x .- circshift(grid.x, (1, 0))
    dy = grid.y .- circshift(grid.y, (0, 1))
    dz = missing
    ApplyDerivativeOperator{T}(GridData(dx), GridData(dy), GridData(dz))
end

struct GridDerivatives{X,Y,Z} <: AbstractGridDerivatives
    dx :: X
    dy :: Y
    dz :: Z
end 
function GridDerivatives(grid::Grid; Kx=1, Ky=1, Kz=1)
    dx = grid.x .- circshift(grid.x, (1, 0))
    dy = grid.y .- circshift(grid.y, (0, 1))
    dz = missing
    GridDerivatives(GridData(dx), GridData(dy), GridData(dz))
end

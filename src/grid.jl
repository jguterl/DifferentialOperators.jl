# a generic uniform grid generator just for testing ... 
Base.@kwdef struct Grid{X,Y,Z} <: AbstractGrid
    x::X = missing
    y::Y = missing
    z::Z = missing
end
const Grid1D{X} = Grid{X,Missing,Missing}
const Grid2D{X,Y} = Grid{X,Y,Missing}
const Grid3D{X,Y,Z} = Grid{X,Y,Z}

#Grid(dims::NTuple{N,Int64}; L=[1.0, 1.0, 1.0], ng = [1, 1, 1], d0=[0.0, 0.0, 0.0]) where {N} = Grid(; (fn => (d0[i] + L[i]) / (dims[i] - 1) * (getindex.(collect(Iterators.product((1:d for d in dims)...)), i) .- 1) for ((i, d), fn) in zip(enumerate(dims), fieldnames(Grid)))...)
Grid(dims::NTuple{N,Int64}; L=[1.0, 1.0, 1.0], ng = [0, 0, 0], d0=[0.0, 0.0, 0.0]) where {N} = Grid(; (fn => (d0[i] + L[i]) / (dims[i] - 1) * (getindex.(collect(Iterators.product((1-ng[i]:d+ng[i] for d in dims)...)), i) .- 1) for ((i, d), fn) in zip(enumerate(dims), fieldnames(Grid)))...)
Grid(nx::Int64, ny::Int64; kw...) = Grid((nx,ny); kw...)
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

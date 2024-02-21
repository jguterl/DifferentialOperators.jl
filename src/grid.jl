# a generic uniform grid generator just for testing ... 

#
# This entire class is just 200 lines of gibberish to provide dx and dy which are constants!
# Down to 100
#

#
# Should change the name of the object, since it collides with Base!!
#
Base.@kwdef struct Grid{X,Y,Z} <: AbstractGrid
    x::X = missing
    y::Y = missing
    z::Z = missing
end

Base.ones(grid::Grid)                    = ones(grid,current_backend.value)
Base.ones(grid::Grid, backend::Backend)  = backend(ones(size(grid.x)...))
Base.zeros(grid::Grid)                   = zeros(grid, current_backend.value)
Base.zeros(grid::Grid, backend::Backend) = backend(zeros(size(grid.x)...))


_get_grid_points(i::Int64, dims, ng, d0, L) = (d0[i] + L[i]) / (dims[i] - 1) * (getindex.(collect(Iterators.product((1-ng[i]:d+ng[i] for d in dims)...)), i) .- 1)
get_grid_points(backend::CPUBackend , args..., )  = _get_grid_points(args...)
get_grid_points(backend::CUDABackend, args..., )  = CUDA.CuArray(_get_grid_points(args...))

#
# The backend should default to CPU if nothing is set
#
Grid(dims::NTuple{N,Int64}; L=[1.0, 1.0, 1.0], ng=[0, 0, 0], d0=[0.0, 0.0, 0.0], backend::Backend=current_backend.value) where {N} = Grid(; (fn => get_grid_points(backend, i, dims, ng, d0, L) for ((i, d), fn) in zip(enumerate(dims), fieldnames(Grid)))...)

Grid(nx::Int64, ny::Int64; kw...) = Grid((nx,ny); kw...)

Base.size(grid::Grid) = size(grid.x)

#
# Schedule this entire class for deletion whenever possible
#
# Why is there another extra parameter V that is not used?
#
struct GridDerivatives{X<:GridData,Y<:GridData,Z<:GridData,B} <: AbstractGridDerivatives{B}
    dx :: X
    dy :: Y
    dz :: Z
    backend::B
end

#function GridDerivatives(grid::Grid, ghost_cells::GhostCells; Kx=1, Ky=1, Kz=1)
function GridDerivatives(grid::Grid) #, ghost_cells::GhostCells; Kx=1, Ky=1, Kz=1)
    dx = grid.x .- circshift(grid.x, (1, 0))
    dy = grid.y .- circshift(grid.y, (0, 1))
    dz = missing
#    set_dx_ghost_cells!(dx, ghost_cells)
#    set_dy_ghost_cells!(dy, ghost_cells)
    GridDerivatives(GridData(dx), GridData(dy), GridData(dz))
end

GridDerivatives(x::AbstractGridData{B}, y::AbstractGridData{B}, z::AbstractGridData{B}) where {B<:Backend} = GridDerivatives(x,y,z,B())
#GridDerivatives(x::GridData{B}, y::GridData{B}, z::GridData{B}) where {B<:Backend} = GridDerivatives(x,y,z,B())

#
# The solution would be for dx to be stored in the grid
#
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

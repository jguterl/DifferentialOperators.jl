

Base.@kwdef struct LogicalCoords{X,Y,Z} <: AbstractGrid
    x::X = missing
    y::Y = missing
    z::Z = missing
end

Base.ones(coords::LogicalCoords)                    = ones(coords,current_backend.value)
Base.ones(coords::LogicalCoords, backend::Backend)  = backend(ones(size(coords.x)...))
Base.zeros(coords::LogicalCoords)                   = zeros(coords, current_backend.value)
Base.zeros(coords::LogicalCoords, backend::Backend) = backend(zeros(size(coords.x)...))


function _get_grid_points(i::Int64, dims, ng, d0, L)
   return (d0[i] + L[i]) / (dims[i] - 1) * ( getindex.(collect(Iterators.product((1-ng[i]:d+ng[i] for d in dims)...)), i) .- 1 )
end

function get_grid_points(backend::CPUBackend , args..., )
    return _get_grid_points(args...)
end

function get_grid_points(backend::CUDABackend, args..., )
    return CUDA.CuArray(_get_grid_points(args...))
end

#
# The backend should default to CPU if nothing is set
#
function LogicalCoords(dims::NTuple{N,Int64}; L=[1.0, 1.0, 1.0], ng=[0, 0, 0], d0=[0.0, 0.0, 0.0], backend::Backend=current_backend.value) where {N}
    return LogicalCoords(; (fn => get_grid_points(backend, i, dims, ng, d0, L) for ((i, d), fn) in zip(enumerate(dims), fieldnames(LogicalCoords)))...)
end
    
function LogicalCoords(nx::Int64, ny::Int64; kw...)
    return LogicalCoords((nx,ny); kw...)
end

Base.size(coords::LogicalCoords) = size(coords.x)

#
# Schedule this entire class for deletion whenever possible
#
struct GridDerivatives{X<:GridData,Y<:GridData,Z<:GridData,B} <: AbstractGridDerivatives{B}
    dx :: X
    dy :: Y
    dz :: Z
    backend::B
end

#
# Get rid of this entire class, for now hack to get the correct spacings -- uniform everywhere
#
#function GridDerivatives(coords::Grid, ghost_cells::GhostCells; Kx=1, Ky=1, Kz=1)
function GridDerivatives(coords::LogicalCoords) #, ghost_cells::GhostCells; Kx=1, Ky=1, Kz=1)
    dx = 0*coords.x .+ ( coords.x[2,1] - coords.x[1,1] ) #coords.x .- circshift(coords.x, (1, 0))
    dy = 0*coords.y .+ ( coords.y[1,2].- coords.x[1,1] ) #coords.y .- circshift(coords.y, (0, 1))
    dz = missing
#    set_dx_ghost_cells!(dx, ghost_cells)
#    set_dy_ghost_cells!(dy, ghost_cells)
    GridDerivatives(GridData(dx), GridData(dy), GridData(dz))
end

GridDerivatives(x::AbstractGridData{B}, y::AbstractGridData{B}, z::AbstractGridData{B}) where {B<:Backend} = GridDerivatives(x,y,z,B())

#
# The solution would be for dx to be stored in the grid
#
get_dx(grid_data::GridDerivatives) = CUDA.@allowscalar grid_data.dx.data[3, 3] #TODO: this is wrong if  ng c >2 

get_dy(grid_data::GridDerivatives) = CUDA.@allowscalar grid_data.dy.data[3, 3] # TODO: this is wrong if  ng c >2 

Adapt.@adapt_structure GridDerivatives

export get_dx, get_dy

#
# Logical coordinate structure
#
Base.@kwdef struct LogicalCoords{X,Y,Z} <: AbstractGrid
    x::X = missing
    y::Y = missing
    z::Z = missing
end

#
# Default constructors
#
function LogicalCoords(dims::NTuple{N,Int64}; L=[1.0, 1.0, 1.0], ng=[0, 0, 0], d0=[0.0, 0.0, 0.0], backend::Backend=current_backend.value) where {N}
    return LogicalCoords(; (fn => get_grid_points(backend, i, dims, ng, d0, L) for ((i, d), fn) in zip(enumerate(dims), fieldnames(LogicalCoords)))...)
end
    
function LogicalCoords(nx::Int64, ny::Int64; kw...)
    return LogicalCoords((nx,ny); kw...)
end

#
# Coordinate grids are defined here
#
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
# Some helper functions
#
Base.ones(coords::LogicalCoords)                    = ones(coords,current_backend.value)
Base.ones(coords::LogicalCoords, backend::Backend)  = backend(ones(size(coords.x)...))
Base.zeros(coords::LogicalCoords)                   = zeros(coords, current_backend.value)
Base.zeros(coords::LogicalCoords, backend::Backend) = backend(zeros(size(coords.x)...))
Base.size(coords::LogicalCoords) = size(coords.x)

const Grid1D{X}     = LogicalCoords{X,Missing,Missing}
const Grid2D{X,Y}   = LogicalCoords{X,Y,Missing}
const Grid3D{X,Y,Z} = LogicalCoords{X,Y,Z}

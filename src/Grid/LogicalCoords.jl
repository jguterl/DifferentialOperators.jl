#
# Logical coordinate structure
#
Base.@kwdef struct LogicalCoords{X,Y,Z} <: AbstractGrid
    #This should end up being only one array element
    x::X = missing
    y::Y = missing
    z::Z = missing
end

#
# Default constructors
#
#
# Idea: LogicalCoords should take array limits that come in a iterators
#       It should not allocate any memory because 3D vectors are expensive
#       The constructor should take an array and convert it to tupple instead of
#       messing around with 57 constructors
#

function LogicalCoords(dims::NTuple{N,Int64}; L::Array{Float64}=[1.0 for i in 1:N], ng::Array{Int64}=[0 for i in 1:N], d0::Array{Float64}=[0.0 for i in 1:N], backend::Backend=current_backend.value) where {N}
    #
    # This line is incomprehensible --- break up into different readable lines
    #
    return LogicalCoords(; (fn => get_grid_points(backend, i, dims, ng, d0, L) for ((i, d), fn) in zip(enumerate(dims), fieldnames(LogicalCoords)))...)
end

function LogicalCoords(npt::Array{Int64}; kw...)
    return LogicalCoords( tuple(npt...); kw...)
end

#
# Coordinate grids are defined here
#
function _get_grid_points(i::Int64, npts, ng, d0, L)
   return (d0[i] + L[i]) / (npts[i] - 1) * ( getindex.(collect(Iterators.product((1-ng[i]:d+ng[i] for d in npts)...)), i) .- 1 )
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

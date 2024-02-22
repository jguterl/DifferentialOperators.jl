module DifferentialOperators

include("Backends/backend.jl")
export set_backend! #, check_backend!

export Field, VectorField, ScalarField, TensorField

export compute!, compute_turbo!, compute_threads!

export LogicalCoords, GridDerivatives, GridData

export StructuredGrid

#export AbstractGrid
#export AbstractStructuredGrid

#
# These likely need to be sent to their respective files
#

# data structures related to grid
abstract type AbstractGrid end 
abstract type AbstractStructuredGrid end
abstract type AbstractGridDerivatives{B<:Backend} end 

# data structures for dispatches purposes 
abstract type Field end 
abstract type AbstractTensorField <: Field end
abstract type AbstractVectorField <: Field end 
abstract type AbstractScalarField <: Field end
abstract type AbstractGridData{B<:Backend} end
abstract type AbstractFieldData{B<:Backend} end

# data structures related to operators
abstract type Operator end
abstract type AbstractApplyOperator{O<:Operator} end

# data structures related to vector and scalar components
abstract type AbstractComponent end
struct XComponent <: AbstractComponent end
struct YComponent <: AbstractComponent end
struct ZComponent <: AbstractComponent end
struct ScalarComponent <: AbstractComponent end

# --- get primary type of object ---
get_base_type(T) = get_base_type(typeof(T))
get_base_type(T::DataType) = T.name.wrapper

#
# Structures and functionality for indices and grids
#
include("Grid/Grid.jl")

#
# Structures for scalar, vector, tensor fields
#
include("Fields/Fields.jl")

#
# Vector calculus, cross and dot product, finite differences
#
include("Operators/Operators.jl")


#
# Pointwise evaluation of the operator for vectors and scalars
#

function compute_point!(op::ScalarField, grid_data::AbstractGridDerivatives, v::ScalarField, i::Index, j::Index)
    #@inbounds v.field[i, j] = op.field(grid_data, i, j)
    v.field[i, j] = op.field(grid_data, i, j)
    nothing
end

function compute_point!(op::VectorField, grid_data::AbstractGridDerivatives, v::VectorField, i::Index, j::Index)
        v.x[i, j] = op.x(grid_data, i, j)
        v.y[i, j] = op.y(grid_data, i, j)
        v.z[i, j] = op.z(grid_data, i, j)
    nothing
end

# -------------------------------------------#
#
# GPU backend calculation
#

import CUDA: i32

function compute!(op::Union{TensorField,ScalarField,VectorField}, grid_data::AbstractGridDerivatives{B}, v, i_::IndexIterator, j_::IndexIterator) where {B<:CUDABackend}
    i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32) * blockDim().y + threadIdx().y
    (i < i_.start || i> i_.stop || j < j_.start || j> j_.stop) && return
    compute_point!(op, grid_data, v, i, j)
    nothing
end

#
# CPU backend computation
#
function compute!(op, grid_data::AbstractGridDerivatives{B}, v, i_::IndexIterator, j_::IndexIterator) where {B<:CPUBackend}
    compute!(op, grid_data, v, i_.start:i_.stop, j_.start:j_.stop)
    return nothing
end

function compute!(op::Union{TensorField,ScalarField,VectorField}, grid_data::AbstractGridDerivatives{B}, v, i_::UnitRange, j_::UnitRange) where {B<:CPUBackend}
    for j in j_
        for i in i_
            compute_point!(op, grid_data, v, i, j)
        end
    end
    nothing
end

#
# Threaded computation -- eventually should be a backend
#
function compute_threads!(op, grid_data::AbstractGridDerivatives{B}, v, i_::IndexIterator, j_::IndexIterator) where {B<:CPUBackend}
    compute_threads!(op, grid_data, v, i_.start:i_.stop, j_.start:j_.stop)
    return nothing
end

function compute_threads!(op::Union{TensorField,ScalarField,VectorField}, grid_data::AbstractGridDerivatives{B}, v::VectorField, i_::UnitRange, j_::UnitRange) where {B<:CPUBackend}
    Threads.@threads for j in j_
        for i in i_
            compute_point!(op, grid_data, v, i, j)
        end
    end
    return nothing
end

# function compute!(grid_data::AbstractGridDerivatives{B}) where {B<:CUDABackend}
#     i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
#     j = (blockIdx().y - 1i32) * blockDim().y + threadIdx().y
#     #(i < i_.start || i> i._stop || j < j_.start || j> j._stop) && return
#     #compute!(op, grid_data, v, i, j)
#     nothing
# end

# function (op::VectorField{X,Y,Z})(grid_data::AbstractGridDerivatives, v::VectorField,  i::Int64, j::Int64)  where {X,Y,Z}
#     v.x[i, j] = op.x(grid_data, i, j)
#     v.y[i, j] = op.y(grid_data,i, j)
#     v.z[i, j] = op.z(grid_data,i, j)
#     nothing
# end


end

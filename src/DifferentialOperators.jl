module DifferentialOperators

using LoopVectorization

include("Backends/backend.jl")
export set_backend! #, check_backend!

export CPUBackend, CUDABackend
export Field, VectorField, ScalarField, TensorField

export compute!, compute_turbo!, compute_threads!

export LogicalCoords, AbstractCoordSpacings, CoordSpacings, CoordData

export StructuredGrid

export AbstractGrid
export AbstractStructuredGrid
export AbstractDlData

#
# These likely need to be sent to their respective files
#

# data structures related to grid
abstract type AbstractGrid end 
abstract type AbstractStructuredGrid end
abstract type AbstractCoordSpacings{B<:Backend} end 
abstract type AbstractCoordData{B<:Backend} end
abstract type AbstractDlData{B<:Backend} end

# data structures for dispatches purposes 
abstract type Field end 
abstract type AbstractTensorField <: Field end
abstract type AbstractVectorField <: Field end 
abstract type AbstractScalarField <: Field end
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

function compute_point!(s_out::ScalarField, op_in::ScalarField, GridSpacings::AbstractCoordSpacings, i::Index, j::Index)
    #@inbounds v.field[i, j] = op.field(GridSpacings, i, j)
    s_out.field.data[i, j] = op_in.field(GridSpacings, i, j)
    nothing
end

function compute_point!(v_out::VectorField, op_in::VectorField, GridSpacings::AbstractCoordSpacings, i::Index, j::Index)
        v_out.x[i, j] = op_in.x(GridSpacings, i, j)
        v_out.y[i, j] = op_in.y(GridSpacings, i, j)
        v_out.z[i, j] = op_in.z(GridSpacings, i, j)
    nothing
end

# -------------------------------------------#
#
# GPU backend calculation
#

import CUDA: i32

function compute!(v, op, GridSpacings::AbstractCoordSpacings{B}, i_::IndexIterator, j_::IndexIterator) where {B<:CUDABackend}
    compute!(v, op, GridSpacings, i_.start:i_.stop, j_.start:j_.stop)
    return nothing
end

function compute!(v, op::Union{TensorField,ScalarField,VectorField}, GridSpacings::AbstractCoordSpacings{B}, i_::UnitRange, j_::UnitRange) where {B<:CUDABackend}
    i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32) * blockDim().y + threadIdx().y
    (i < i_.start || i> i_.stop || j < j_.start || j> j_.stop) && return
    compute_point!(v, op, GridSpacings, i, j)
    nothing
end

function compute!(v, op::Union{TensorField,ScalarField,VectorField}, GridSpacings::AbstractCoordSpacings{B}, i_::IndexIterator, j_::IndexIterator) where {B<:CUDABackend}
    i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32) * blockDim().y + threadIdx().y
    (i < i_.start || i> i_.stop || j < j_.start || j> j_.stop) && return
    compute_point!(v, op, GridSpacings, i, j)
    nothing
end

#
# CPU backend computation
#
function compute!(v, op, StructuredGrid::AbstractStructuredGrid)
    compute!(v, op, StructuredGrid.Spacings, StructuredGrid.InteriorPoints[1], StructuredGrid.InteriorPoints[2])
    return nothing
end

function compute!(v, op, GridSpacings::AbstractCoordSpacings{B}, i_::IndexIterator, j_::IndexIterator) where {B<:CPUBackend}
    compute!(v, op, GridSpacings, i_.start:i_.stop, j_.start:j_.stop)
    return nothing
end

function compute!(v, op::Union{TensorField,ScalarField,VectorField}, GridSpacings::AbstractCoordSpacings{B}, i_::UnitRange, j_::UnitRange) where {B<:CPUBackend}
    for j in j_
        for i in i_
            compute_point!(v, op, GridSpacings, i, j)
        end
    end
    nothing
end



#
# Threaded computation -- eventually should be a backend
#
function compute_threads!(v, op, StructuredGrid::AbstractStructuredGrid)
    compute_threads!(v, op, StructuredGrid.GridSpacings, StructuredGrid.InteriorPoints[1], StructuredGrid.InteriorPoints[2])
    return nothing
end

function compute_threads!(v, op, GridSpacings::AbstractCoordSpacings{B}, i_::IndexIterator, j_::IndexIterator) where {B<:CPUBackend}
    compute_threads!(v, op, GridSpacings, i_.start:i_.stop, j_.start:j_.stop)
    return nothing
end

function compute_threads!(v, op::Union{TensorField,ScalarField,VectorField}, GridSpacings::AbstractCoordSpacings{B}, i_::UnitRange, j_::UnitRange) where {B<:CPUBackend}
    Threads.@threads for j in j_
        for i in i_
            compute_point!(v, op, GridSpacings, i, j)
        end
    end
    return nothing
end

# function compute!(GridSpacings::AbstractCoordSpacings{B}) where {B<:CUDABackend}
#     i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
#     j = (blockIdx().y - 1i32) * blockDim().y + threadIdx().y
#     #(i < i_.start || i> i._stop || j < j_.start || j> j._stop) && return
#     #compute!(op, GridSpacings, v, i, j)
#     nothing
# end

# function (op::VectorField{X,Y,Z})(GridSpacings::AbstractCoordSpacings, v::VectorField,  i::Int64, j::Int64)  where {X,Y,Z}
#     v.x[i, j] = op.x(GridSpacings, i, j)
#     v.y[i, j] = op.y(GridSpacings, i, j)
#     v.z[i, j] = op.z(GridSpacings, i, j)
#     nothing
# end


end

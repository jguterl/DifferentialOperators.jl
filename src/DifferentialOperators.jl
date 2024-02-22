module DifferentialOperators

include("backend.jl")
export set_backend!
export check_backend!

abstract type AbstractGrid end 
abstract type AbstractMHDGrid end
abstract type AbstractGridDerivatives{B<:Backend} end 



# data structures for dispatches purposes 
abstract type Field end 
abstract type AbstractTensorField <: Field end
abstract type AbstractVectorField <: Field end 
abstract type AbstractScalarField <: Field end
abstract type AbstractGridData{B<:Backend} end
abstract type AbstractFieldData{B<:Backend} end
abstract type Operator end
abstract type AbstractComponent end
abstract type AbstractApplyOperator{O<:Operator} end
struct XComponent <: AbstractComponent end
struct YComponent <: AbstractComponent end
struct ZComponent <: AbstractComponent end
struct ScalarComponent <: AbstractComponent end
Index = Union{Int64,Int32}
export AbstractGrid, AbstractMHDGrid

# --- get primary type of object ---
get_base_type(T) = get_base_type(typeof(T))
get_base_type(T::DataType) = T.name.wrapper


include("Fields.jl")
include("Operators.jl")
include("grid.jl")
include("junk.jl")
#include("FiniteDifferences.jl")
#include("Products.jl")

#
# Be careful with the INBOUNDS that remove check of out of bound indexes
function compute!(op::ScalarField, grid_data::AbstractGridDerivatives, v::ScalarField, i::Int64, j::Int64)
    @inbounds v.field[i, j] = op.field(grid_data, i, j)
    nothing
end

function compute!(op::VectorField, grid_data::AbstractGridDerivatives, v::VectorField, i::Index, j::Index)
        @inbounds v.x[i, j] = op.x(grid_data, i, j)
        @inbounds v.y[i, j] = op.y(grid_data, i, j)
        @inbounds v.z[i, j] = op.z(grid_data, i, j)
    nothing
end

# -------------------------------------------#
import CUDA: i32
function compute!(op::Union{TensorField,ScalarField,VectorField}, grid_data::AbstractGridDerivatives{B}, v, i_::IndexIterator, j_::IndexIterator) where {B<:CUDABackend}
    i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32) * blockDim().y + threadIdx().y
    (i < i_.start || i> i_.stop || j < j_.start || j> j_.stop) && return
    compute!(op, grid_data, v, i, j)
    nothing
end

# function compute!(grid_data::AbstractGridDerivatives{B}) where {B<:CUDABackend}
#     i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
#     j = (blockIdx().y - 1i32) * blockDim().y + threadIdx().y
#     #(i < i_.start || i> i._stop || j < j_.start || j> j._stop) && return
#     #compute!(op, grid_data, v, i, j)
#     nothing
# end

compute!(op, grid_data::AbstractGridDerivatives{B}, v, i_::IndexIterator, j_::IndexIterator) where {B<:CPUBackend} = compute!(op, grid_data, v, i_.start:i_.stop, j_.start:j_.stop)
function compute!(op::Union{TensorField,ScalarField,VectorField}, grid_data::AbstractGridDerivatives{B}, v, i_::UnitRange, j_::UnitRange) where {B<:CPUBackend}
    for j in j_
        for i in i_
            compute!(op, grid_data, v, i, j)
        end
    end
    nothing
end

compute_threads!(op, grid_data::AbstractGridDerivatives{B}, v, i_::IndexIterator, j_::IndexIterator) where {B<:CPUBackend} = compute_threads!(op, grid_data, v, i_.start:i_.stop, j_.start:j_.stop)
function compute_threads!(op::Union{TensorField,ScalarField,VectorField}, grid_data::AbstractGridDerivatives{B}, v::VectorField, i_::UnitRange, j_::UnitRange) where {B<:CPUBackend}
    Threads.@threads for j in j_
                        for i in i_
                            compute!(op, grid_data, v, i, j)
                        end
    end
    nothing
end


export compute!, compute_turbo!, compute_threads!

# function (op::VectorField{X,Y,Z})(grid_data::AbstractGridDerivatives, v::VectorField,  i::Int64, j::Int64)  where {X,Y,Z}
#     v.x[i, j] = op.x(grid_data, i, j)
#     v.y[i, j] = op.y(grid_data,i, j)
#     v.z[i, j] = op.z(grid_data,i, j)
#     nothing
# end




export MHDGrid, StructuredGrid, VectorField, GridDerivatives, ScalarField, TensorField, Field, GridData
end

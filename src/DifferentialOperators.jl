module DifferentialOperators

include("backend.jl")

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

export AbstractGrid, AbstractMHDGrid

# --- get primary type of object ---
get_base_type(T) = get_base_type(typeof(T))
get_base_type(T::DataType) = T.name.wrapper


include("fields.jl")


struct ApplyOperator{D,V,O<:Operator,C<:AbstractComponent} <: AbstractApplyOperator{O}
    data::D
    var::V
    o::O
    component::C
end

Adapt.@adapt_structure TensorField

ApplyOperatorX(d, v, o) = ApplyOperator(d, v, o, XComponent())
ApplyOperatorY(d, v, o) = ApplyOperator(d, v, o, YComponent())
ApplyOperatorZ(d, v, o) = ApplyOperator(d, v, o, ZComponent())
ApplyOperatorScalar(d, v, o) = ApplyOperator(d, v, o, ScalarComponent())
export ApplyOperator
function compute!(op::VectorField{X,Y,Z},grid_data::AbstractGridDerivatives, v::VectorField, i::Int64, j::Int64) where {X,Y,Z}
    v.x[i, j] = op.x(grid_data, i, j)
    v.y[i, j] = op.y(grid_data, i, j)
    v.z[i, j] = op.z(grid_data, i, j)
    nothing
end
function evaluate!(op::VectorField{X,Y,Z}, grid_data::AbstractGridDerivatives, v::VectorField, i::Int64, j::Int64) where {X,Y,Z}
    v.x[i, j] = op.x(grid_data, i, j)
    v.y[i, j] = op.y(grid_data, i, j)
    v.z[i, j] = op.z(grid_data, i, j)
    nothing
end
function compute!(op::ScalarField, grid_data::AbstractGridDerivatives, v::ScalarField, i_::UnitRange{Int64}, j_::UnitRange{Int64}) 
    @inbounds for j in j_
        for i in i_
            v.field[i, j] = op.field(grid_data, i, j)
        end
    end
    nothing
end

function compute!(op::ScalarField, grid_data::AbstractGridDerivatives, v::ScalarField, i_::UnitRange{Int64}, j_::UnitRange{Int64})
    @inbounds for j in j_
        for i in i_
            v.field[i, j] = op.field(grid_data, i, j)
        end
    end
    nothing
end

#
# This should be a scalar to vector transformation
#
function compute!(op::ScalarField, grid_data::AbstractGridDerivatives, v::VectorField, i::Int64, j::Int64)
            @inbounds v.x[i, j] = op.x(grid_data, i, j)
            @inbounds v.y[i, j] = op.y(grid_data, i, j)
            @inbounds v.z[i, j] = op.z(grid_data, i, j)
    nothing
end


function compute!(op::ScalarField, grid_data::AbstractGridDerivatives{B}, v::VectorField, i_::UnitRange{Int64}, j_::UnitRange{Int64}) where {B<:CPUBackend}
    i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1i32) * blockDim().y + threadIdx().y
    (i < i_.start || i> i._stop || j < j_.start || j> j._stop) && return
    compute!(op, grid_data, v, i, j)
    end
    nothing
end

function compute!(op::ScalarField, grid_data::AbstractGridDerivatives{B}, v::VectorField, i_::UnitRange{Int64}, j_::UnitRange{Int64}) where {B<:CUDABackend}
    @cuda for j in j_
        for i in i_
            compute!(op, grid_data, v, i, j)
        end
    end
    nothing
end


function compute!(op::VectorField{X,Y,Z}, grid_data::AbstractGridDerivatives{B}, v::VectorField, i_::UnitRange{Int64}, j_::UnitRange{Int64}) where {X,Y,Z,B<:CPUBackend}
    @inbounds for j in j_
        for i in i_
        v.x[i, j] = op.x(grid_data, i, j)
        v.y[i, j] = op.y(grid_data, i, j)
        v.z[i, j] = op.z(grid_data, i, j)
        end
    end
    nothing
end

function _compute!(vx,vy,vz,opx,opy, opz, grid_data, i, j)
            vx[i, j] = opx(grid_data, i, j)
            vy[i, j] = opy(grid_data, i, j)
            vz[i, j] = opz(grid_data, i, j)
end
function compute_turbo!(op::VectorField{X,Y,Z}, grid_data::AbstractGridDerivatives, v::VectorField, i_::UnitRange{Int64}, j_::UnitRange{Int64}) where {X,Y,Z}
    # @turbo 
    for j in j_
        for i in i_
        _compute!(v.x,v.y,v.z,op.x,op.y, op.z, grid_data, i, j)
    end
end
    nothing
end

# function compute_turbo!(op::VectorField{X,Y,Z}, grid_data::AbstractGridDerivatives, v::VectorField, i_::UnitRange{Int64}, j_::UnitRange{Int64}) where {X,Y,Z}
#      for j in j_
#         for i in i_
#             @inbounds begin
#                 v.x[i, j] = op.x(grid_data, i, j) # note: possible to completely fuse that loop as well but derivative at boundaries need to be handle properly
#                 v.y[i, j] = op.y(grid_data, i, j)
#                 v.z[i, j] = op.z(grid_data, i, j)
#             end
#         end
#     end
#     nothing
# end

function compute_threads!(op::VectorField{X,Y,Z}, grid_data::AbstractGridDerivatives, v::VectorField, i_::UnitRange{Int64}, j_::UnitRange{Int64}) where {X,Y,Z}
    Threads.@threads for j in j_
                            for i in i_
                                @inbounds begin
                                    v.x[i, j] = op.x(grid_data, i, j) # note: possible to completely fuse that loop as well but derivative at boundaries need to be handle properly
                                    v.y[i, j] = op.y(grid_data, i, j)
                                    v.z[i, j] = op.z(grid_data, i, j)
                                end
                            end
    end
    nothing
end


export compute!, compute_turbo!, compute_threads!

function (op::VectorField{X,Y,Z})(grid_data::AbstractGridDerivatives, v::VectorField,  i::Int64, j::Int64)  where {X,Y,Z}
    v.x[i, j] = op.x(grid_data, i, j)
    v.y[i, j] = op.y(grid_data,i, j)
    v.z[i, j] = op.z(grid_data,i, j)
    nothing
end


include("operators.jl")
include("grid.jl")
include("derivatives.jl")
export MHDGrid, Grid, VectorField, GridDerivatives, ScalarField, TensorField, Field
end

module DifferentialOperators
using LoopVectorization

abstract type AbstractGrid end 
abstract type AbstractGridDerivatives end 

# data structures for dispatches purposes 
abstract type AbstractVectorField end 
abstract type AbstractScalarField end
abstract type AbstractGridData{T} end
abstract type AbstractFieldData{T} end

struct GridData{T} <: AbstractGridData{T}
    data::T
end

struct FieldData{T} <: AbstractFieldData{T}
    data::T
end

# accesssor
# Horrible hack warning: the grid spacings look wrong near the edge. This is OK for uniform grid at the moment
(d::GridData)(i, j) = d.data[i, j]

(d::FieldData{T})(grid_data::AbstractGridDerivatives,i, j) where T = d.data[i, j]
(d::FieldData{T})(i, j) where {T} = d.data[i, j]


#generic setor (could add typing if concern with dimension compability) 
Base.setindex!(f::FieldData, args...) = setindex!(f.data, args...)
Base.getindex(v::FieldData, args...) = getindex(v.data, args...)

#
# Vector field structure
#
# we only consider 3 components because of curl 
struct VectorField{X,Y,Z} <: AbstractVectorField
    x::X 
    y::Y 
    z::Z 
end

#constructors
VectorField(dims::NTuple{N,Int64}) where {N} = VectorField((FieldData(zeros(dims...)) for fn in fieldnames(VectorField))...);
VectorField(n::Int64) = VectorField((n,))
VectorField(nx::Int64, ny::Int64) = VectorField((nx,ny))
VectorField(grid::AbstractGrid) = VectorField(size(grid))

#
# Scalar field structure
#
struct ScalarField{D} <: AbstractScalarField
    field::D
end

#generic setor (could add typing if concern with dimension compability) 
#ScalarField(dims::NTuple{N,Int64}) where {N} = ScalarField((zeros(dims...) for fn in fieldnames(ScalarField))...)
ScalarField(dims::NTuple{N,Int64}) where {N} = ScalarField((FieldData(zeros(dims...)) for fn in fieldnames(ScalarField))...)
ScalarField(n::Int64) = ScalarField((n,))
ScalarField(nx::Int64, ny::Int64) = ScalarField((nx, ny))
ScalarField(grid::AbstractGrid) = ScalarField(size(grid))


abstract type Operator end


abstract type AbstractComponent end
abstract type AbstractApplyOperator{O<:Operator} end
struct XComponent <: AbstractComponent end
struct YComponent <: AbstractComponent end
struct ZComponent <: AbstractComponent end
struct ScalarComponent <: AbstractComponent end
struct ApplyOperator{D,V,O<:Operator,C<:AbstractComponent} <: AbstractApplyOperator{O}
    data::D
    var::V
    o::O
    component::C
end
ApplyOperatorX(d, v, o) = ApplyOperator(d, v, o, XComponent())
ApplyOperatorY(d, v, o) = ApplyOperator(d, v, o, YComponent())
ApplyOperatorZ(d, v, o) = ApplyOperator(d, v, o, ZComponent())
ApplyOperatorScalar(d, v, o) = ApplyOperator(d, v, o, ScalarComponent())

function compute!(op::VectorField{X,Y,Z},grid_data::AbstractGridDerivatives, v::VectorField, i::Int64, j::Int64) where {X,Y,Z}
    v.x[i, j] = op.x(grid_data, i, j)
    v.y[i, j] = op.y(grid_data, i, j)
    v.z[i, j] = op.z(grid_data, i, j)
    nothing
end

#
# Scalar to scalar single threaded operation
#
function compute!(op::ScalarField, grid_data::AbstractGridDerivatives, v::ScalarField, i_::UnitRange{Int64}, j_::UnitRange{Int64}) 
    @inbounds for j in j_
        for i in i_
            v.field[i, j] = op.field(grid_data, i, j)
        end
    end
    nothing
end

#
# Scalar to scalar single threaded operation
#
function compute_threads!(op::ScalarField, grid_data::AbstractGridDerivatives, v::ScalarField, i_::UnitRange{Int64}, j_::UnitRange{Int64}) 
    Threads.@threads for j in j_
        for i in i_
            @inbounds begin
                v.field[i, j] = op.field(grid_data, i, j)
            end
        end
    end
    nothing
end

#
# Scalar to vector single threaded operation
#
function compute!(op::ScalarField, grid_data::AbstractGridDerivatives, v::VectorField, i_::UnitRange{Int64}, j_::UnitRange{Int64}) 
    @inbounds for j in j_
        for i in i_
            v.x[i, j] = op.x(grid_data, i, j)
            v.y[i, j] = op.y(grid_data, i, j)
            v.z[i, j] = op.z(grid_data, i, j)
        end
    end
    nothing
end

#
# Scalar to vector multi-threaded operation
#
function compute_threads!(op::ScalarField, grid_data::AbstractGridDerivatives, v::VectorField, i_::UnitRange{Int64}, j_::UnitRange{Int64}) 
    Threads.@threads for j in j_
        for i in i_
            @inbounds begin
                v.x[i, j] = op.x(grid_data, i, j)
                v.y[i, j] = op.y(grid_data, i, j)
                v.z[i, j] = op.z(grid_data, i, j)
            end
        end
    end
    nothing
end

#
# Vector to vector single threaded operation
#
function compute!(op::VectorField{X,Y,Z}, grid_data::AbstractGridDerivatives, v::VectorField, i_::UnitRange{Int64}, j_::UnitRange{Int64}) where {X,Y,Z}
    @inbounds for j in j_
        for i in i_
        v.x[i, j] = op.x(grid_data, i, j)
        v.y[i, j] = op.y(grid_data, i, j)
        v.z[i, j] = op.z(grid_data, i, j)
        end
    end
    nothing
end

#
# Vector to vector multi-threaded operation
#
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

function _compute!(vx,vy,vz,opx,opy,opz,grid_data, i, j)
            vx[i, j] = opx(grid_data, i, j)
            vy[i, j] = opy(grid_data, i, j)
            vz[i, j] = opz(grid_data, i, j)
end
function compute_turbo!(op::VectorField{X,Y,Z}, grid_data::AbstractGridDerivatives, v::VectorField, i_::UnitRange{Int64}, j_::UnitRange{Int64}) where {X,Y,Z}
    #@turbo 
    for j in j_
        for i in i_
            _compute!(v.x,v.y,v.z,op.x,op.y,op.z,grid_data,i,j)
        end
    end
    nothing
end

function compute_turbo!(op::VectorField{X,Y,Z}, grid_data::AbstractGridDerivatives, v::VectorField, i_::UnitRange{Int64}, j_::UnitRange{Int64}) where {X,Y,Z}
     for j in j_
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
export Grid, VectorField, GridDerivatives, ScalarField
end

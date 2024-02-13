module DifferentialOperators
abstract type AbstractGrid end 
abstract type AbstractGridDerivatives end 

# data structures for dispatches purposes 
abstract type AbstractVectorField end 
abstract type AbstractGridData{T} end
abstract type AbstractFieldData{T} end

struct GridData{T} <: AbstractGridData{T}
    data::T
end

(dx::GridData)(i::Int64, j::Int64) = dx.data[i, j]
struct FieldData{T} <: AbstractFieldData{T}
    data::T
end
# accesssor
(g::GridData)(i::Int64, j::Int64)= g.data[i, j]
(d::FieldData{T})(grid_data::AbstractGridDerivatives,i, j) where T = d.data[i, j]
(d::FieldData{T})(i, j) where {T} = d.data[i, j]
(g::GridData{T})(i) where T<:Array{1}= g.data[i, j]
(d::FieldData{T})(i) where T<:Array{1} = d.data[i, j]

#generic setor (could add typing if concern with dimension compability) 
Base.setindex!(f::FieldData, args...) = setindex!(f.data, args...)

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

abstract type Operator end


abstract type AbstractComponent end
abstract type AbstractApplyOperator{O<:Operator} end
struct XComponent <: AbstractComponent end
struct YComponent <: AbstractComponent end
struct ZComponent <: AbstractComponent end
struct ApplyOperator{D,V,O<:Operator,C<:AbstractComponent} <: AbstractApplyOperator{O}
    data::D
    var::V
    o::O
    component::C
end
ApplyOperatorX(d, v, o) = ApplyOperator(d, v, o, XComponent())
ApplyOperatorY(d, v, o) = ApplyOperator(d, v, o, YComponent())
ApplyOperatorZ(d, v, o) = ApplyOperator(d, v, o,ZComponent())

function compute!(op::VectorField{X,Y,Z},grid_data::AbstractGridDerivatives, v::VectorField, i::Int64, j::Int64) where {X,Y,Z}
    v.x[i, j] = op.x(grid_data, i, j)
    v.y[i, j] = op.y(grid_data, i, j)
    v.z[i, j] = op.z(grid_data, i, j)
    nothing
end

function compute!(op::VectorField{X,Y,Z}, grid_data::AbstractGridDerivatives, v::VectorField, i_::UnitRange{Int64}, j_::UnitRange{Int64}) where {X,Y,Z}
    for j in j_
        for i in i_
        v.x[i, j] = op.x(grid_data, i, j)
        v.y[i, j] = op.y(grid_data, i, j)
        v.z[i, j] = op.z(grid_data, i, j)
        end
    end
    nothing
end
export compute!

function (op::VectorField{X,Y,Z})(grid_data::AbstractGridDerivatives, v::VectorField,  i::Int64, j::Int64)  where {X,Y,Z}
    v.x[i, j] = op.x(grid_data, i, j)
    v.y[i, j] = op.y(grid_data,i, j)
    v.z[i, j] = op.z(grid_data,i, j)
    nothing
end


include("operators.jl")
include("grid.jl")
export Grid, VectorField, GridDerivatives
end

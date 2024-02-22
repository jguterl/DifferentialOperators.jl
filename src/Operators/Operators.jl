
struct ApplyOperator{D, V, O <: Operator, C <: AbstractComponent} <: AbstractApplyOperator{O}
    data::D
    var::V
    o::O
    component::C
end

#
# Here we need extensive breaking apart into files and rearrangement
#

ApplyOperatorX(d, v, o) = ApplyOperator(d, v, o, XComponent())
ApplyOperatorY(d, v, o) = ApplyOperator(d, v, o, YComponent())
ApplyOperatorZ(d, v, o) = ApplyOperator(d, v, o, ZComponent())
ApplyOperatorScalar(d, v, o) = ApplyOperator(d, v, o, ScalarComponent())
Adapt.@adapt_structure ApplyOperator
export ApplyOperator

abstract type AbstractOperator{D,V,O<:Operator} end

include("FiniteDifferences.jl")
include("CenteredGradients.jl")
include("ForwardStaggeredGradients.jl")
include("BackwardStaggeredGradients.jl")
include("Products.jl")

struct GenericOperator{T}  <: Operator end
GenericOperator(T) = GenericOperator{T}()

# generic operator
#
# What the hell is this??? CHange the name O to something else, can be confused with 0, really bad practice to use a single char
# anywhere except indices
#
(op::ApplyOperator{D,V,GenericOperator{O},XComponent})(args...) where {O,D<:VectorField,V<:VectorField} = O(op.var.x(args...), op.data.x(args...))
(op::ApplyOperator{D,V,GenericOperator{O},YComponent})(args...) where {O,D<:VectorField,V<:VectorField} = O(op.var.y(args...), op.data.y(args...))
(op::ApplyOperator{D,V,GenericOperator{O},ZComponent})(args...) where {O,D<:VectorField,V<:VectorField} = O(op.var.z(args...), op.data.z(args...))
(op::ApplyOperator{D,V,GenericOperator{O},ScalarComponent})(args...) where {O,D<:ScalarField,V<:ScalarField} = O(op.var.field(args...), op.data.field(args...))
(op::ApplyOperator{D,V,GenericOperator{O},XComponent})(args...) where {O,D<:Float64    ,V<:VectorField}  = O(op.var.x(args...), op.data)
(op::ApplyOperator{D,V,GenericOperator{O},YComponent})(args...) where {O,D<:Float64    ,V<:VectorField}  = O(op.var.y(args...), op.data)
(op::ApplyOperator{D,V,GenericOperator{O},ZComponent})(args...) where {O,D<:Float64    ,V<:VectorField}  = O(op.var.z(args...), op.data)
(op::ApplyOperator{D,V,GenericOperator{O},XComponent})(args...) where {O,D<:ScalarField,V<:VectorField}  = O(op.var.x(args...), op.data.field(args...),)
(op::ApplyOperator{D,V,GenericOperator{O},YComponent})(args...) where {O,D<:ScalarField,V<:VectorField}  = O(op.var.y(args...), op.data.field(args...),)
(op::ApplyOperator{D,V,GenericOperator{O},ZComponent})(args...) where {O,D<:ScalarField,V<:VectorField}  = O(op.var.Z(args...), op.data.field(args...),)
(op::ApplyOperator{D,V,GenericOperator{O},ScalarComponent})(args...) where {O,D<:Float64,V<:ScalarField} = O(op.var.field(args...), op.data)



VectorField(d, v, o::Operator)  = VectorField(ApplyOperatorX(d, v, o), ApplyOperatorY(d, v, o), ApplyOperatorZ(d, v, o))
ScalarField(d, v, o::Operator)  = ScalarField(ApplyOperatorScalar(d, v, o))

#
# What is the wiggly product
#
export ∇, ∇², ∇⁺, ∇⁻, ×, ⋅

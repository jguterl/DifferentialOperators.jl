
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

#Centered operators
struct CurlOperator        <: Operator end
struct GradientOperator    <: Operator end
struct DivergenceOperator  <: Operator end
struct LaplacianOperator   <: Operator end

#Forward staggered
struct Curl⁺Operator       <: Operator end
struct Gradient⁺Operator   <: Operator end
struct Divergence⁺Operator <: Operator end

#Backward staggered
struct Curl⁻Operator       <: Operator end
struct Gradient⁻Operator   <: Operator end
struct Divergence⁻Operator <: Operator end

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


#
# Centered operators
#

# curl --- vector into vector
(op::ApplyOperator{D,V,CurlOperator,XComponent})(args...) where {D,V} = (∂y(op.var.z, args...) - ∂z(op.var.y, args...))
(op::ApplyOperator{D,V,CurlOperator,YComponent})(args...) where {D,V} = (∂z(op.var.x, args...) - ∂x(op.var.z, args...))
(op::ApplyOperator{D,V,CurlOperator,ZComponent})(args...) where {D,V} = (∂x(op.var.y, args...) - ∂y(op.var.x, args...))

# gradient --- scalar into vector
(op::ApplyOperator{D,V,GradientOperator,XComponent})(args...) where {D,V} = ∂x(op.var.field, args...)
(op::ApplyOperator{D,V,GradientOperator,YComponent})(args...) where {D,V} = ∂y(op.var.field, args...)
(op::ApplyOperator{D,V,GradientOperator,ZComponent})(args...) where {D,V} = ∂z(op.var.field, args...)

# divergence --- vector into scalar
(op::ApplyOperator{D,V,DivergenceOperator,ScalarComponent})(args...) where {D,V} = ∂x(op.var.x, args...) + ∂y(op.var.y, args...) + ∂z(op.var.z, args...)

# laplacian --- scalar into scalar
(op::ApplyOperator{D,V,LaplacianOperator,ScalarComponent})(args...) where {D,V} = ∂x²(op.var.field, args...) + ∂y²(op.var.field, args...) + ∂z²(op.var.field, args...)

#
# Forward staggered
#

# curl+ --- vector into vector
(op::ApplyOperator{D,V,Curl⁺Operator,XComponent})(args...) where {D,V} = (∂y⁺(op.var.z, args...) - ∂z⁺(op.var.y, args...))
(op::ApplyOperator{D,V,Curl⁺Operator,YComponent})(args...) where {D,V} = (∂z⁺(op.var.x, args...) - ∂x⁺(op.var.z, args...))
(op::ApplyOperator{D,V,Curl⁺Operator,ZComponent})(args...) where {D,V} = (∂x⁺(op.var.y, args...) - ∂y⁺(op.var.x, args...))

# gradient+ --- scalar into vector
(op::ApplyOperator{D,V,Gradient⁺Operator,XComponent})(args...) where {D,V} = ∂x⁺(op.var.field, args...)
(op::ApplyOperator{D,V,Gradient⁺Operator,YComponent})(args...) where {D,V} = ∂y⁺(op.var.field, args...)
(op::ApplyOperator{D,V,Gradient⁺Operator,ZComponent})(args...) where {D,V} = ∂z⁺(op.var.field, args...)

#divergence --- vector into scalar
(op::ApplyOperator{D,V,Divergence⁺Operator,ScalarComponent})(args...) where {D,V} = ∂x⁺(op.var.x, args...) + ∂y⁺(op.var.y, args...) + ∂z⁺(op.var.z, args...)

#
# Backward staggered
#

# curl-
(op::ApplyOperator{D,V,Curl⁻Operator,XComponent})(args...) where {D,V} = (∂y⁻(op.var.z, args...) - ∂z⁻(op.var.y, args...))
(op::ApplyOperator{D,V,Curl⁻Operator,YComponent})(args...) where {D,V} = (∂z⁻(op.var.x, args...) - ∂x⁻(op.var.z, args...))
(op::ApplyOperator{D,V,Curl⁻Operator,ZComponent})(args...) where {D,V} = (∂x⁻(op.var.y, args...) - ∂y⁻(op.var.x, args...))

# gradient-
(op::ApplyOperator{D,V,Gradient⁻Operator,XComponent})(args...) where {D,V} = ∂x⁻(op.var.field, args...)
(op::ApplyOperator{D,V,Gradient⁻Operator,YComponent})(args...) where {D,V} = ∂y⁻(op.var.field, args...)
(op::ApplyOperator{D,V,Gradient⁻Operator,ZComponent})(args...) where {D,V} = ∂z⁻(op.var.field, args...)

# divergence --- vector into scalar
(op::ApplyOperator{D,V,Divergence⁻Operator,ScalarComponent})(args...) where {D,V} = ∂x⁻(op.var.x, args...) + ∂y⁻(op.var.y, args...) + ∂z⁻(op.var.z, args...)

abstract type AbstractOperator{D,V,O<:Operator} end 
#Centered operators
Curl{D,V}          = AbstractOperator{D,V,CurlOperator}
Gradient{D,V}      = AbstractOperator{D,V,GradientOperator}
∇{D,V}             = AbstractOperator{D,V,GradientOperator}
∇(v::ScalarField)  = VectorField(nothing, v, GradientOperator())
×(::Type{∇} , v::VectorField) = VectorField(nothing, v, CurlOperator()) #non-commutative operator
⋅(::Type{∇}, var::VectorField) = ScalarField(nothing, var, DivergenceOperator())#non-commutative operator

∇²{D,V}            = AbstractOperator{D,V,LaplacianOperator}
∇²(v::ScalarField) = ScalarField(nothing, v, LaplacianOperator())

#Forward staggered operators
Curl⁺{D,V}         = AbstractOperator{D,V,Curl⁺Operator}
Gradient⁺{D,V}     = AbstractOperator{D,V,Gradient⁺Operator}
∇⁺{D,V}            = AbstractOperator{D,V,Gradient⁺Operator}
∇⁺(v::ScalarField) = VectorField(nothing, v, Gradient⁺Operator())
×(::Type{∇⁺}, vec::VectorField) = VectorField(nothing, vec, Curl⁺Operator()) #non-commutative operator
⋅(::Type{∇⁺}, vec::VectorField) = ScalarField(nothing, vec, Divergence⁺Operator())#non-commutative operator

#Backward staggered operators
Curl⁻{D,V}         = AbstractOperator{D,V,Curl⁻Operator}
Gradient⁻{D,V}     = AbstractOperator{D,V,Gradient⁻Operator}
∇⁻{D,V}            = AbstractOperator{D,V,Gradient⁻Operator}
∇⁻(scal::ScalarField) = VectorField(nothing, scal, Gradient⁻Operator())
×(::Type{∇⁻}, vec::VectorField) = VectorField(nothing, vec, Curl⁻Operator())#non-commutative operator
⋅(::Type{∇⁻}, vec::VectorField) = ScalarField(nothing, vec, Divergence⁻Operator())#non-commutative operator

#Product{D,V}  = AbstractOperator{D,V,ProductOperator}

VectorField(d, v, o::Operator)  = VectorField(ApplyOperatorX(d, v, o), ApplyOperatorY(d, v, o), ApplyOperatorZ(d, v, o))
ScalarField(d, v, o::Operator)  = ScalarField(ApplyOperatorScalar(d, v, o))

#
# What is the wiggly product
#
export ∇, ∇², ∇⁺, ∇⁻, ×, ⋅

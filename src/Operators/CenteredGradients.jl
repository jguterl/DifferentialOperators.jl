#Centered operators
struct CurlOperator        <: Operator end
struct GradientOperator    <: Operator end
struct DivergenceOperator  <: Operator end
struct LaplacianOperator   <: Operator end

#
# Centered operators
#
Curl{D,V}          = AbstractOperator{D,V,CurlOperator}
Gradient{D,V}      = AbstractOperator{D,V,GradientOperator}
∇{D,V}             = AbstractOperator{D,V,GradientOperator}
∇(s::ScalarField)  = VectorField(nothing, s, GradientOperator())

#These products define curl and div
×(::Type{∇}, vec::VectorField) = VectorField(nothing, vec, CurlOperator()) #non-commutative operator
⋅(::Type{∇}, vec::VectorField) = ScalarField(nothing, vec, DivergenceOperator())#non-commutative operator

∇²{D,V}            = AbstractOperator{D,V,LaplacianOperator}
∇²(s::ScalarField) = ScalarField(nothing, s, LaplacianOperator())

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



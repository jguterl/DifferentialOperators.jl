#Forward staggered
struct Curl⁺Operator       <: Operator end
struct Gradient⁺Operator   <: Operator end
struct Divergence⁺Operator <: Operator end

#
# Forward staggered
#
Curl⁺{D,V}         = AbstractOperator{D,V,Curl⁺Operator}
Gradient⁺{D,V}     = AbstractOperator{D,V,Gradient⁺Operator}
∇⁺{D,V}            = AbstractOperator{D,V,Gradient⁺Operator}
∇⁺(s::ScalarField) = VectorField(nothing, s, Gradient⁺Operator())

#These products define curl and div
×(::Type{∇⁺}, vec::VectorField) = VectorField(nothing, vec, Curl⁺Operator()) #non-commutative operator
⋅(::Type{∇⁺}, vec::VectorField) = ScalarField(nothing, vec, Divergence⁺Operator())#non-commutative operator

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

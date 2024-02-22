#Product
# struct ProductOperator     <: Operator end #not needed
struct ScalarProductOperator <: Operator end #TODO:where does it belong -> center?
#struct ContractionOperator   <: Operator end #TODO:where does it belong -> center?
struct CrossProductOperator  <: Operator end #TODO:where does it belong -> center?

# # multiply
# (op::ApplyOperator{D,V,ProductOperator,XComponent})(args...) where {D<:Float64,V} = op.var.x(args...) * op.data
# (op::ApplyOperator{D,V,ProductOperator,YComponent})(args...) where {D<:Float64,V} = op.var.y(args...) * op.data
# (op::ApplyOperator{D,V,ProductOperator,ZComponent})(args...) where {D<:Float64,V} = op.var.z(args...) * op.data

# scalar product
(op::ApplyOperator{D,V,ScalarProductOperator,ScalarComponent})(args...) where {D<:VectorField,V<:VectorField} = op.var.x(args...) * op.data.x(args...) + op.var.y(args...) * op.data.y(args...) + op.var.z(args...) * op.data.z(args...)

# crossproduct
(op::ApplyOperator{D,V,CrossProductOperator,XComponent})(args...) where {D<:VectorField,V<:VectorField} = op.var.y(args...) * op.data.z(args...) - op.var.z(args...) * op.data.y(args...) 
(op::ApplyOperator{D,V,CrossProductOperator,YComponent})(args...) where {D<:VectorField,V<:VectorField} = op.var.z(args...) * op.data.x(args...) - op.var.x(args...) * op.data.z(args...)
(op::ApplyOperator{D,V,CrossProductOperator,ZComponent})(args...) where {D<:VectorField,V<:VectorField} = op.var.x(args...) * op.data.y(args...) - op.var.y(args...) * op.data.x(args...)

×(a::T, b::U) where {T<:Float64,U<:Field} = get_base_type(U)(a, b, GenericOperator(*)) #commutative operator
×(a::U, b::T) where {T<:Float64,U<:Field} = get_base_type(U)(b, a, GenericOperator(*)) #commutative operator

#TODO: Check dot and cross product. 
#TODO: Verify commutativity! 

 # #non-commutative operator
#does not exist!!!!
#×(b::VectorField, a::ScalarField) = VectorField(a, b, ProductOperator()) #commutative operator 
×(a::ScalarField, b::VectorField) = VectorField(a, b, ProductOperator()) #commutative operator
×(a::VectorField, b::VectorField) = VectorField(b, a, CrossProductOperator()) #non-commutative operator

#scalar product 
⋅(a::VectorField, b::VectorField) = ScalarField(b, a, ScalarProductOperator()) #commutative operator

# ∂t operator
import Base: +, * ,/, -
Base.:+(a::T, b::U) where {U<:Union{Field,Float64},T<:Field} = get_base_type(T)(b, a, GenericOperator(+))
Base.:-(a::T, b::U) where {U<:Union{Field,Float64},T<:Field} = get_base_type(T)(b, a, GenericOperator(-))
Base.:*(a::T, b::U) where {U<:Union{Field,Float64},T<:Field} = get_base_type(T)(b, a, GenericOperator(*))
Base.:/(a::T, b::U) where {U<:Union{Field,Float64},T<:Field} = get_base_type(T)(b, a, GenericOperator(/))

Base.:+(a::T, b::U) where {T<:Union{Float64},U<:Field} = get_base_type(U)(b, a, GenericOperator(+))
Base.:-(a::T, b::U) where {T<:Union{Float64},U<:Field} = get_base_type(U)(b, a, GenericOperator(-))
Base.:*(a::T, b::U) where {T<:Union{Float64},U<:Field} = get_base_type(U)(b, a, GenericOperator(*))
Base.:/(a::T, b::U) where {T<:Union{Float64},U<:Field} = get_base_type(U)(b, a, GenericOperator(/))

#export ×, ⋅, ∻

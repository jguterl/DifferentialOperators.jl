
struct ApplyOperator{D, V, O <: Operator, C <: AbstractComponent} <: AbstractApplyOperator{O}
    data::D
    var::V
    o::O
    component::C
end



ApplyOperatorX(d, v, o) = ApplyOperator(d, v, o, XComponent())
ApplyOperatorY(d, v, o) = ApplyOperator(d, v, o, YComponent())
ApplyOperatorZ(d, v, o) = ApplyOperator(d, v, o, ZComponent())
ApplyOperatorScalar(d, v, o) = ApplyOperator(d, v, o, ScalarComponent())
Adapt.@adapt_structure ApplyOperator
export ApplyOperator


#Product
# struct ProductOperator     <: Operator end #not needed
struct ScalarProductOperator      <: Operator end #TODO:where does it belong -> center?
struct ContractionOperator <: Operator end #TODO:where does it belong -> center?
struct CrossProductOperator <: Operator end #TODO:where does it belong -> center?

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
(op::ApplyOperator{D,V,GenericOperator{O},XComponent})(args...) where {O,D<:VectorField,V<:VectorField} = O(op.var.x(args...), op.data.x(args...))
(op::ApplyOperator{D,V,GenericOperator{O},YComponent})(args...) where {O,D<:VectorField,V<:VectorField} = O(op.var.y(args...), op.data.y(args...))
(op::ApplyOperator{D,V,GenericOperator{O},ZComponent})(args...) where {O,D<:VectorField,V<:VectorField} = O(op.var.z(args...), op.data.z(args...))
(op::ApplyOperator{D,V,GenericOperator{O},ScalarComponent})(args...) where {O,D<:ScalarField,V<:ScalarField} = O(op.var.field(args...), op.data.field(args...))
(op::ApplyOperator{D,V,GenericOperator{O},XComponent})(args...) where {O,D<:Float64,V<:VectorField} = O(op.var.x(args...), op.data)
(op::ApplyOperator{D,V,GenericOperator{O},YComponent})(args...) where {O,D<:Float64,V<:VectorField} = O(op.var.y(args...), op.data)
(op::ApplyOperator{D,V,GenericOperator{O},ZComponent})(args...) where {O,D<:Float64,V<:VectorField} = O(op.var.z(args...), op.data)
(op::ApplyOperator{D,V,GenericOperator{O},XComponent})(args...) where {O,D<:ScalarField,V<:VectorField} = O(op.var.x(args...), op.data.field(args...),)
(op::ApplyOperator{D,V,GenericOperator{O},YComponent})(args...) where {O,D<:ScalarField,V<:VectorField} = O(op.var.y(args...), op.data.field(args...),)
(op::ApplyOperator{D,V,GenericOperator{O},ZComponent})(args...) where {O,D<:ScalarField,V<:VectorField} = O(op.var.Z(args...), op.data.field(args...),)
(op::ApplyOperator{D,V,GenericOperator{O},ScalarComponent})(args...) where {O,D<:Float64,V<:ScalarField} = O(op.var.field(args...), op.data)

# # multiply
# (op::ApplyOperator{D,V,ProductOperator,XComponent})(args...) where {D<:Float64,V} = op.var.x(args...) * op.data
# (op::ApplyOperator{D,V,ProductOperator,YComponent})(args...) where {D<:Float64,V} = op.var.y(args...) * op.data
# (op::ApplyOperator{D,V,ProductOperator,ZComponent})(args...) where {D<:Float64,V} = op.var.z(args...) * op.data

# scalar product
(op::ApplyOperator{D,V,ScalarProductOperator,ScalarComponent})(args...) where {D<:VectorField,V<:VectorField} = op.var.x(args...) * op.data.x(args...) + op.var.y(args...) * op.data.y(args...) + op.var.z(args...) * op.data.z(args...)

#contraction product
(op::ApplyOperator{D,V,ContractionOperator,XComponent})(args...) where {D<:VectorField,V<:TensorField} = op.var.xx(args...) * op.data.x(args...) + op.var.xy(args...) * op.data.y(args...) + op.var.xz(args...) * op.data.z(args...)
(op::ApplyOperator{D,V,ContractionOperator,YComponent})(args...) where {D<:VectorField,V<:TensorField} = op.var.yx(args...) * op.data.x(args...) + op.var.yy(args...) * op.data.y(args...) + op.var.yz(args...) * op.data.z(args...)
(op::ApplyOperator{D,V,ContractionOperator,ZComponent})(args...) where {D<:VectorField,V<:TensorField} = op.var.zx(args...) * op.data.x(args...) + op.var.zy(args...) * op.data.y(args...) + op.var.zz(args...) * op.data.z(args...)

# crossproduct
(op::ApplyOperator{D,V,CrossProductOperator,XComponent})(args...) where {D<:VectorField,V<:VectorField} = op.var.y(args...) * op.data.z(args...) - op.var.z(args...) * op.data.y(args...) 
(op::ApplyOperator{D,V,CrossProductOperator,YComponent})(args...) where {D<:VectorField,V<:VectorField} = op.var.z(args...) * op.data.x(args...) - op.var.x(args...) * op.data.z(args...)
(op::ApplyOperator{D,V,CrossProductOperator,ZComponent})(args...) where {D<:VectorField,V<:VectorField} = op.var.x(args...) * op.data.y(args...) - op.var.y(args...) * op.data.x(args...)

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
# This looks like ( diag ( grad ( V ) ) )
# (op::ApplyOperator{D,V,GradientOperator,XComponent})(args...) where {D,V} = ∂x(op.var.x, args...)
# (op::ApplyOperator{D,V,GradientOperator,YComponent})(args...) where {D,V} = ∂y(op.var.y, args...)
# (op::ApplyOperator{D,V,GradientOperator,ZComponent})(args...) where {D,V} = ∂z(op.var.z, args...)

# divergence --- vector into scalar
(op::ApplyOperator{D,V,DivergenceOperator,ScalarComponent})(args...) where {D,V} = ∂x(op.var.x, args...) + ∂y(op.var.y, args...) + ∂z(op.var.z, args...)

# laplacian --- scalar into scalar
(op::ApplyOperator{D,V,LaplacianOperator,ScalarComponent})(args...) where {D,V} = ∂x²(op.var.field, args...) + ∂y²(op.var.field, args...) + ∂z²(op.var.field, args...)
#(op::ApplyOperator{D,V,LaplacianOperator,YComponent})(args...) where {D,V} = ∂x²(op.var.y, args...) + ∂y²(op.var.y, args...) + ∂z²(op.var.y, args...)
#(op::ApplyOperator{D,V,LaplacianOperator,ZComponent})(args...) where {D,V} = ∂x²(op.var.z, args...) + ∂y²(op.var.z, args...) + ∂z²(op.var.z, args...)

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
∇²{D,V}            = AbstractOperator{D,V,LaplacianOperator}
∇²(v::ScalarField) = ScalarField(nothing, v, LaplacianOperator())


#∇(v::TensorVectorField) = VectorField(nothing, v, GradientOperator())

# ---- time operator-----
abstract type Apply∂ₜ{C} end 

function Apply∂ₜ{C}(Δt::Float64, v :: Union{Field,ApplyOperator}, order) where {C}
    order == 1 && return Apply∂ₜ1order{C}(Δt, v)
    error("not implemented yet....")
    order == 2 && return Apply∂ₜ2order{C}(Δt, v)
end

∂ₜ(v::T, Δt::Float64, order=1) where {T<:Union{Field,ApplyOperator}} = get_base_type(T)((Apply∂ₜ{fn}(Δt, v, order) for fn in fieldnames(T))...)

struct Apply∂ₜ1order{C,F<:Vector,T<:Union{Field,ApplyOperator}} <: Apply∂ₜ{C}
    Δt::F
    var::T
    var_old::T
end

Apply∂ₜ1order{C}(Δt::Float64, v::T) where {C,T} = Apply∂ₜ1order{C,Vector{Float64},T}([Δt], v,copy(v))

struct Apply∂ₜ2order{C,F<:Vector,T<:Union{Field,ApplyOperator}} <: Apply∂ₜ{C}# let's not use mutable struct because of gpu
    Δt::F 
    var::T
    var_old::T
    var_old2::T
end

Apply∂ₜ2order{C}(Δt::Float64, v::T) where {C,T} = Apply∂ₜ2order{C,Vector{Float64},T}([Δt, Δt], v, copy(v), copy(v))

set_dt!(∂ₜ::Apply∂ₜ1order, dt::Float64) = ∂ₜ.Δt[1] = dt

(∂ₜ::Apply∂ₜ1order{:x,F,T})(args...) where {F,T}= (∂ₜ.var.x(args...) - ∂ₜ.var_old.x(args...)) / Δt[1]
(∂ₜ::Apply∂ₜ1order{:y,F,T})(args...) where {F, T} = (∂ₜ.var.y(args...) - ∂ₜ.var_old.y(args...)) / Δt[1]
(∂ₜ::Apply∂ₜ1order{:z,F,T})(args...) where {F, T} = (∂ₜ.var.z(args...) - ∂ₜ.var_old.z(args...)) / Δt[1]
(∂ₜ::Apply∂ₜ1order{:field,F,T})(args...) where {F, T} = (∂ₜ.var.field(args...) - ∂ₜ.var_old.field(args...)) / Δt[1]

(∂ₜ::Apply∂ₜ2order)(args...) = error() # TODO .... 

# ---------------------------------------------- #
#
# What are these constructors(?) for????? I added the gradient ones, but they don't seem to 
# do anything
#
#Gradient(v)       = Gradient(nothing, v) ???
#Gradient(d,v)     = VectorField(d, v, GradientOperator()) ???

#Forward staggered operators
Curl⁺{D,V}         = AbstractOperator{D,V,Curl⁺Operator}
Gradient⁺{D,V}     = AbstractOperator{D,V,Gradient⁺Operator}
∇⁺{D,V}            = AbstractOperator{D,V,Gradient⁺Operator}
∇⁺(v::ScalarField) = VectorField(nothing, v, Gradient⁺Operator())

#Backward staggered operators

Gradient⁻{D,V}     = AbstractOperator{D,V,Gradient⁻Operator}
∇⁻{D,V}            = AbstractOperator{D,V,Gradient⁻Operator}
∇⁻(v::ScalarField) = VectorField(nothing, v, Gradient⁻Operator())


#
# Product operations defining ∇⋅, ∇×, scalar product
#
#Product{D,V}  = AbstractOperator{D,V,ProductOperator}

#TODO: Check dot and cross product. 
#TODO: Verify commutativity! 

×(::Type{∇}, v::VectorField) = VectorField(nothing, v, CurlOperator()) #non-commutative operator
×(::Type{∇⁺}, v::VectorField) = VectorField(nothing, v, Curl⁺Operator()) #non-commutative operator
×(::Type{∇⁻}, v::VectorField) = VectorField(nothing, v, Curl⁻Operator())#non-commutative operator
×(a::T, b::U) where {T<:Float64,U<:Field} = get_base_type(U)(a, b, GenericOperator(*)) #commutative operator
×(a::U, b::T) where {T<:Float64,U<:Field} = get_base_type(U)(b, a, GenericOperator(*)) #commutative operator

 # #non-commutative operator
×(a::ScalarField, b::VectorField) = VectorField(a, b, ProductOperator()) #commutative operator
×(b::VectorField, a::ScalarField) = VectorField(a, b, ProductOperator()) #commutative operator
×(a::VectorField, b::VectorField) = VectorField(b, a, CrossProductOperator()) #non-commutative operator

⋅(::Type{∇}, var::VectorField) = ScalarField(nothing, var, DivergenceOperator())#non-commutative operator
⋅(::Type{∇⁺}, var::VectorField) = ScalarField(nothing, var, Divergence⁺Operator())#non-commutative operator
⋅(::Type{∇⁻}, var::VectorField) = ScalarField(nothing, var, Divergence⁻Operator())#non-commutative operator

#scalar product 
⋅(a::VectorField, b::VectorField) = ScalarField(b, a, ScalarProductOperator()) #commutative operator
⋅(a::TensorField, b::VectorField) = VectorField(b, a, ContractionOperator())  #non-commutative operator
∻(a::TensorField, b::VectorField) = ScalarField(b, ⋅(a, b), ScalarProductOperator()) #non-commutative operator #notation: ∻ = `\kernelcontraction` 
#TODO: FH check contraction

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

VectorField(d, v, o::Operator)  = VectorField(ApplyOperatorX(d, v, o), ApplyOperatorY(d, v, o), ApplyOperatorZ(d, v, o))
ScalarField(d, v, o::Operator)  = ScalarField(ApplyOperatorScalar(d, v, o))

export ∇, ∇², ∇⁺, ∇⁻, ×, ⋅, ∻, ∂ₜ

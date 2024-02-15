#Product
struct ProductOperator    <: Operator end

#Centered operators
struct CurlOperator       <: Operator end
struct GradientOperator   <: Operator end
struct LaplacienOperator  <: Operator end
struct DivergenceOperator <: Operator end

#Forward staggered
struct Curl⁺Operator      <: Operator end
struct Gradient⁺Operator  <: Operator end

#Backward staggered
struct Curl⁻Operator      <: Operator end
struct Gradient⁻Operator  <: Operator end

# multiply
(op::ApplyOperator{D,V,ProductOperator,XComponent})(args...) where {D<:Float64,V} = op.var.x(args...) * op.data
(op::ApplyOperator{D,V,ProductOperator,YComponent})(args...) where {D<:Float64,V} = op.var.y(args...) * op.data
(op::ApplyOperator{D,V,ProductOperator,ZComponent})(args...) where {D<:Float64,V} = op.var.z(args...) * op.data

#
# Centered operators
#

#curl
(op::ApplyOperator{D,V,CurlOperator,XComponent})(args...) where {D,V} = (∂y(op.var.z, args...) - ∂z(op.var.y, args...))
(op::ApplyOperator{D,V,CurlOperator,YComponent})(args...) where {D,V} = (∂z(op.var.x, args...) - ∂x(op.var.z, args...))
(op::ApplyOperator{D,V,CurlOperator,ZComponent})(args...) where {D,V} = (∂x(op.var.y, args...) - ∂y(op.var.x, args...))

#gradient --- warning, this should work for scalars, what does op.var contain?? Right now this is like diag( grad( V ) )
(op::ApplyOperator{D,V,GradientOperator,XComponent})(args...) where {D,V} = ∂x(op.var.field, args...)
(op::ApplyOperator{D,V,GradientOperator,YComponent})(args...) where {D,V} = ∂y(op.var.field, args...)
(op::ApplyOperator{D,V,GradientOperator,ZComponent})(args...) where {D,V} = ∂z(op.var.field, args...)
#(op::ApplyOperator{D,V,GradientOperator,XComponent})(args...) where {D,V} = ∂x(op.var.x, args...)
#(op::ApplyOperator{D,V,GradientOperator,YComponent})(args...) where {D,V} = ∂y(op.var.y, args...)
#(op::ApplyOperator{D,V,GradientOperator,ZComponent})(args...) where {D,V} = ∂z(op.var.z, args...)

#
# Forward staggered
#

#curl+
(op::ApplyOperator{D,V,Curl⁺Operator,XComponent})(args...) where {D,V} = (∂y⁺(op.var.z, args...) - ∂z⁺(op.var.y, args...))
(op::ApplyOperator{D,V,Curl⁺Operator,YComponent})(args...) where {D,V} = (∂z⁺(op.var.x, args...) - ∂x⁺(op.var.z, args...))
(op::ApplyOperator{D,V,Curl⁺Operator,ZComponent})(args...) where {D,V} = (∂x⁺(op.var.y, args...) - ∂y⁺(op.var.x, args...))

#gradient+ --- warning, this should work for scalars, what does op.var contain?? Right now this is like diag( grad( V ) )
(op::ApplyOperator{D,V,Gradient⁺Operator,XComponent})(args...) where {D,V} = ∂x⁺(op.var.x, args...)
(op::ApplyOperator{D,V,Gradient⁺Operator,YComponent})(args...) where {D,V} = ∂y⁺(op.var.y, args...)
(op::ApplyOperator{D,V,Gradient⁺Operator,ZComponent})(args...) where {D,V} = ∂z⁺(op.var.z, args...)

#
# Backward staggered
#

#curl-
(op::ApplyOperator{D,V,Curl⁻Operator,XComponent})(args...) where {D,V} = (∂y⁻(op.var.z, args...) - ∂z⁻(op.var.y, args...))
(op::ApplyOperator{D,V,Curl⁻Operator,YComponent})(args...) where {D,V} = (∂z⁻(op.var.x, args...) - ∂x⁻(op.var.z, args...))
(op::ApplyOperator{D,V,Curl⁻Operator,ZComponent})(args...) where {D,V} = (∂x⁻(op.var.y, args...) - ∂y⁻(op.var.x, args...))

#gradient- --- warning, this should work for scalars, what does op.var contain?? Right now this is like diag( grad( V ) )
(op::ApplyOperator{D,V,Gradient⁻Operator,XComponent})(args...) where {D,V} = ∂x⁻(op.var.x, args...)
(op::ApplyOperator{D,V,Gradient⁻Operator,YComponent})(args...) where {D,V} = ∂y⁻(op.var.y, args...)
(op::ApplyOperator{D,V,Gradient⁻Operator,ZComponent})(args...) where {D,V} = ∂z⁻(op.var.z, args...)

#
# Laplacian and divergence
#

#laplacien --- TODO: eventually this should overloaded as psi = div-( grad+( phi ) or A = - curl-( curl+ ( B ) + grad-(div+( B ) )
#this routine is actually not hooked up to anything that is exported
(op::ApplyOperator{D,V,LaplacienOperator,XComponent})(args...) where {D,V} = ∂x²(op.var.x, args...) + ∂y²(op.var.x, args...) + ∂z²(op.var.x, args...)
(op::ApplyOperator{D,V,LaplacienOperator,YComponent})(args...) where {D,V} = ∂x²(op.var.y, args...) + ∂y²(op.var.y, args...) + ∂z²(op.var.y, args...)
(op::ApplyOperator{D,V,LaplacienOperator,ZComponent})(args...) where {D,V} = ∂x²(op.var.z, args...) + ∂y²(op.var.z, args...) + ∂z²(op.var.z, args...)

#divergence
(op::ApplyOperator{D,V,DivergenceOperator,ScalarComponent})(args...) where {D,V} = ∂x(op.var.x, args...) + ∂y(op.var.y, args...) + ∂z(op.var.z, args...)

# TODO: dispatch on derivative accuracy orders

#Centered differences
∂x(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.5 * (v(i + 1, j) - v(i - 1, j)) / grid_data.dx(i, j)
∂y(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.5 * (v(i, j + 1) - v(i, j - 1)) / grid_data.dy(i, j)
∂z(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0

#Forward staggered
∂x⁺(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (v(i + 1, j) - v(i, j)) / grid_data.dx(i, j)
∂y⁺(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (v(i, j + 1) - v(i, j)) / grid_data.dy(i, j)
∂z⁺(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0

#Backward staggered
∂x⁻(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (v(i, j) - v(i - 1, j)) / grid_data.dx(i, j)
∂y⁻(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (v(i, j) - v(i, j - 1)) / grid_data.dy(i, j)
∂z⁻(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0

# This looks like operations for stretched grids -- I'll just replicate for ∂x⁺, ∂x⁻, etc but I'll ask JG
∂x(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.5 * (op(grid_data, i + 1, j) - op(grid_data, i - 1, j)) / grid_data.dx(i, j)
∂y(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.5 * (op(grid_data, i, j + 1) - op(grid_data, i, j - 1)) / grid_data.dy(i, j)
∂z(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.0

∂x⁺(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (op(grid_data, i + 1, j) - op(grid_data, i, j)) / grid_data.dx(i, j)
∂y⁺(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (op(grid_data, i, j + 1) - op(grid_data, i, j)) / grid_data.dy(i, j)
∂z⁺(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.0

∂x⁻(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (op(grid_data, i, j) - op(grid_data, i - 1, j)) / grid_data.dx(i, j)
∂y⁻(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (op(grid_data, i, j) - op(grid_data, i, j - 1)) / grid_data.dy(i, j)
∂z⁻(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.0

abstract type AbstractOperator{D,V,O<:Operator} end 
#Centered operators
Curl{D,V}         = AbstractOperator{D,V,CurlOperator}
Gradient{D,V}     = AbstractOperator{D,V,GradientOperator}
∇{D,V}            = AbstractOperator{D,V,GradientOperator}
∇(v::ScalarField) = VectorField(nothing, v, GradientOperator())
#∇(v::VectorField) = VectorField(nothing, v, GradientOperator())


Curl(v)           = Curl(nothing, v)
Curl(d, v)        = VectorField(d, v, CurlOperator())


#Forward stagged operators
Curl⁺{D,V}         = AbstractOperator{D,V,Curl⁺Operator}
Gradient⁺{D,V}     = AbstractOperator{D,V,Gradient⁺Operator}
∇⁺{D,V}            = AbstractOperator{D,V,Gradient⁺Operator}
∇⁺(v::VectorField) = VectorField(nothing, v, Gradient⁺Operator())

Curl⁺(v)           = Curl⁺(nothing, v)
Curl⁺(d, v)        = VectorField(d, v, Curl⁺Operator())

#Backward stagged operators
Curl⁻{D,V}         = AbstractOperator{D,V,Curl⁻Operator}
Gradient⁻{D,V}     = AbstractOperator{D,V,Gradient⁻Operator}
∇⁻{D,V}            = AbstractOperator{D,V,Gradient⁻Operator}
∇⁻(v::VectorField) = VectorField(nothing, v, Gradient⁻Operator())

Curl⁻(v)           = Curl⁻(nothing, v)
Curl⁻(d, v)        = VectorField(d, v, Curl⁻Operator())

#
# Product operations defining ∇⋅, ∇×, scalar product
#
Product{D,V}  = AbstractOperator{D,V,ProductOperator}


×( ::Type{∇} , v ::VectorField) = Curl(v)
×( ::Type{∇⁺}, v ::VectorField) = Curl⁺(v)
×( ::Type{∇⁻}, v ::VectorField) = Curl⁻(v)
×(s::Float64, var::VectorField) = VectorField(s, var, ProductOperator())
⋅( ::Type{∇}, var::VectorField) = ScalarField(nothing, var, DivergenceOperator())

VectorField(d, v, o::Operator)  = VectorField(ApplyOperatorX(d, v, o), ApplyOperatorY(d, v, o), ApplyOperatorZ(d, v, o))
ScalarField(d, v, o::Operator)  = ScalarField(ApplyOperatorScalar(d, v, o))

export ∇, ∇⁺, ∇⁻, ×, ⋅

struct CurlOperator <: Operator end
struct GradientOperator <: Operator end
struct ProductOperator <: Operator end
struct LaplacienOperator <: Operator end
struct DivergenceOperator <: Operator end 
# multiply
(op::ApplyOperator{D,V,ProductOperator,XComponent})(args...) where {D<:Float64,V} = op.var.x(args...) * op.data
(op::ApplyOperator{D,V,ProductOperator,YComponent})(args...) where {D<:Float64,V} = op.var.y(args...) * op.data
(op::ApplyOperator{D,V,ProductOperator,ZComponent})(args...) where {D<:Float64,V} = op.var.z(args...) * op.data

#curl
(op::ApplyOperator{D,V,CurlOperator,XComponent})(args...) where {D,V} = (∂y(op.var.z, args...) - ∂z(op.var.y, args...))
(op::ApplyOperator{D,V,CurlOperator,YComponent})(args...) where {D,V} = (∂x(op.var.z, args...) - ∂z(op.var.x, args...))
(op::ApplyOperator{D,V,CurlOperator,ZComponent})(args...) where {D,V} = (∂x(op.var.y, args...) - ∂y(op.var.x, args...))

#gradient
(op::ApplyOperator{D,V,GradientOperator,XComponent})(args...) where {D,V} = ∂x(op.var.x, args...)
(op::ApplyOperator{D,V,GradientOperator,YComponent})(args...) where {D,V} = ∂z(op.var.y, args...)
(op::ApplyOperator{D,V,GradientOperator,ZComponent})(args...) where {D,V} = ∂z(op.var.z, args...)

#laplacien
(op::ApplyOperator{D,V,LaplacienOperator,XComponent})(args...) where {D,V} = ∂x²(op.var.x, args...) + ∂y²(op.var.x, args...) + ∂z²(op.var.x, args...)
(op::ApplyOperator{D,V,LaplacienOperator,YComponent})(args...) where {D,V} = ∂x²(op.var.y, args...) + ∂y²(op.var.y, args...) + ∂z²(op.var.y, args...)
(op::ApplyOperator{D,V,LaplacienOperator,ZComponent})(args...) where {D,V} = ∂x²(op.var.z, args...) + ∂y²(op.var.z, args...) + ∂z²(op.var.z, args...)

#laplacien
(op::ApplyOperator{D,V,DivergenceOperator,ScalarComponent})(args...) where {D,V} = ∂x(op.var.x, args...) + ∂y(op.var.x, args...) + ∂z(op.var.x, args...)

# TODO: dispatch on derivative accuracy orders
∂x(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (v(i + 1, j) - v(i, j)) / grid_data.dx(i, j)
∂y(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (v(i, j + 1) - v(i, j)) / grid_data.dy(i, j)
∂z(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0


∂x(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (op(grid_data, i + 1, j) - op(grid_data, i, j)) / grid_data.dx(i, j)
∂y(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (op(grid_data, i, j + 1) - op(grid_data, i, j)) / grid_data.dy(i, j)
∂z(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.0


abstract type AbstractOperator{D,V,O<:Operator} end 
Curl{D,V} = AbstractOperator{D,V,CurlOperator}
Gradient{D,V} = AbstractOperator{D,V,GradientOperator}
∇{D,V} = AbstractOperator{D,V,GradientOperator}
Product{D,V} = AbstractOperator{D,V,ProductOperator}

∇(v::VectorField) = VectorField(nothing, v, GradientOperator())
Curl(v) = Curl(nothing, v)
Curl(d, v) = VectorField(d, v, CurlOperator())

×(::Type{∇}, v::VectorField) = Curl(v)
×(s::Float64, var::VectorField) = VectorField(s, var, ProductOperator())
⋅(::Type{∇}, var::VectorField) = ScalarField(nothing, var, DivergenceOperator())
VectorField(d, v, o::Operator) = VectorField(ApplyOperatorX(d, v, o), ApplyOperatorY(d, v, o), ApplyOperatorZ(d, v, o))
ScalarField(d, v, o::Operator) = ScalarField(ApplyOperatorScalar(d, v, o))

export ∇, ×, ⋅
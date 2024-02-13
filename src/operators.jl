struct CurlOperator <: Operator end
struct GradientOperator <: Operator end
struct ProductOperator <: Operator end
struct LaplacienOperator <: Operator end
# multiply
(op::ApplyOperator{D,V,ProductOperator,XComponent})(args...) where {D<:Float64,V} = op.var.x(args...) * op.data
(op::ApplyOperator{D,V,ProductOperator,YComponent})(args...) where {D<:Float64,V} = op.var.y(args...) * op.data
(op::ApplyOperator{D,V,ProductOperator,ZComponent})(args...) where {D<:Float64,V} = op.var.z(args...) * op.data

#curl
(op::ApplyOperator{D,V,CurlOperator,XComponent})(args...) where {D,V} = (∂y(op.var.z, args...) - ∂z(op.var.y, args...))
(op::ApplyOperator{D,V,CurlOperator,YComponent})(args...) where {D,V} = (∂x(op.var.z, args...) - ∂z(op.var.x, args...))
(op::ApplyOperator{D,V,CurlOperator,ZComponent})(args...) where {D,V} = (∂x(op.var.z, args...) - ∂z(op.var.x, args...))

#gradient
(op::ApplyOperator{D,V,GradientOperator,XComponent})(args...) where {D,V} = ∂x(op.var.x, args...)
(op::ApplyOperator{D,V,GradientOperator,YComponent})(args...) where {D,V} = ∂z(op.var.y, args...)
(op::ApplyOperator{D,V,GradientOperator,ZComponent})(args...) where {D,V} = ∂z(op.var.z, args...)

#laplacien
(op::ApplyOperator{D,V,LaplacienOperator,XComponent})(args...) where {D,V} = ∂x²(op.var.x, args...) + ∂y²(app.x, args...) + ∂z²(app.x, args...)
(op::ApplyOperator{D,V,LaplacienOperator,YComponent})(args...) where {D,V} = ∂x²(op.var.y, args...) + ∂y²(app.y, args...) + ∂z²(app.y, args...)
(op::ApplyOperator{D,V,LaplacienOperator,ZComponent})(args...) where {D,V} = ∂x²(op.var.z, args...) + ∂y²(app.z, args...) + ∂z²(app.z, args...)

#vector field
#(op::VectorField)(args...) where {D,V} = ∂x²(app.x, args...) + ∂y²(app.x, args...) + ∂z²(app.x, args...)
# (op::ApplyOperator{LaplacienOperator,D,V,YComponent})(args...) where {D,V} = ∂x²(app.y, args...) + ∂y²(app.y, args...) + ∂z²(app.y, args...)
# (op::ApplyOperator{LaplacienOperator,D,V,ZComponent})(args...) where {D,V} = ∂x²(app.z, args...) + ∂y²(app.z, args...) + ∂z²(app.z, args...)

# op_x(var::VectorField, args...) = var.x(args...)
# op_y(var::VectorField, args...) = var.y(args...)
# op_z(var::VectorField, args...) = var.z(args...)
# User-frontend operator definition 
abstract type AbstractOperator{D,V,O<:Operator} end 
Curl{D,V} = AbstractOperator{D,V,CurlOperator}
Gradient{D,V} = AbstractOperator{D,V,GradientOperator}
∇{D,V} = AbstractOperator{D,V,GradientOperator}
Product{D,V} = AbstractOperator{D,V,ProductOperator}

∇(v::VectorField) = VectorField(nothing, v, GradientOperator())
Curl(v) = Curl(nothing, v)
Curl(d, v) = VectorField(d, v, CurlOperator())

# AbstractOperator{D,V,O}(v) where {D,V,O} = AbstractOperator{D,V,O}(nothing,v)
# AbstractOperator{D,V,O}(d, v) where {D,V,O} = AbstractOperator{D,V,O}(d, v, O())
VectorField(d, v, o::Operator) = VectorField(ApplyOperatorX(d, v, o), ApplyOperatorY(d, v, o), ApplyOperatorZ(d, v, o))

#(Applicator{D,V,O})(v) where {D,V,O} = Applicator{D,V,O}(nothing, var, CurlOperator())
×(s::Type{∇}, v::VectorField) = Curl(v)
×(s::Float64, var::VectorField) = VectorField(s, var, ProductOperator())

export ∇, ×
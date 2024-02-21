#contraction product
(op::ApplyOperator{D,V,ContractionOperator,XComponent})(args...) where {D<:VectorField,V<:TensorField} = op.var.xx(args...) * op.data.x(args...) + op.var.xy(args...) * op.data.y(args...) + op.var.xz(args...) * op.data.z(args...)
(op::ApplyOperator{D,V,ContractionOperator,YComponent})(args...) where {D<:VectorField,V<:TensorField} = op.var.yx(args...) * op.data.x(args...) + op.var.yy(args...) * op.data.y(args...) + op.var.yz(args...) * op.data.z(args...)
(op::ApplyOperator{D,V,ContractionOperator,ZComponent})(args...) where {D<:VectorField,V<:TensorField} = op.var.zx(args...) * op.data.x(args...) + op.var.zy(args...) * op.data.y(args...) + op.var.zz(args...) * op.data.z(args...)

⋅(a::TensorField, b::VectorField) = VectorField(b, a, ContractionOperator())  #non-commutative operator
∻(a::TensorField, b::VectorField) = ScalarField(b, ⋅(a, b), ScalarProductOperator()) #non-commutative operator #notation: ∻ = `\kernelcontraction` 
#TODO: FH check contraction

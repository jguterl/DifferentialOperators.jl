#
# TensorField structure
#
struct TensorField{XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ} <: AbstractTensorField
    xx::XX
    xy::XY
    xz::XZ
    yx::YX
    yy::YY
    yz::YZ
    zx::ZX
    zy::ZY
    zz::ZZ
end

#constructors
TensorField(dims::NTuple{N,Int64}) where {N} = TensorField((FieldData(zeros(dims...)) for fn in fieldnames(TensorField))...);
TensorField(n::Int64) = TensorField((n,))
TensorField(nx::Int64, ny::Int64) = TensorField((nx, ny))
TensorField(grid::AbstractGrid) = TensorField(size(grid))
Adapt.@adapt_structure TensorField

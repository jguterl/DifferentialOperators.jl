#
# Vector field structure
#
struct VectorField{X,Y,Z} <: AbstractVectorField
    x::X
    y::Y
    z::Z
end

#constructors
VectorField(dims::NTuple{N,Int64}) where {N} = VectorField((FieldData(zeros(dims...)) for fn in fieldnames(VectorField))...);
VectorField(n::Int64) = VectorField((n,))
VectorField(nx::Int64, ny::Int64) = VectorField((nx, ny))
VectorField(grid::AbstractGrid) = VectorField(size(grid))
VectorField(mhd_grid::AbstractMHDGrid) = VectorField(size(mhd_grid.grid))
Adapt.@adapt_structure VectorField
Base.copy(v::T) where T<:Field= get_base_type(T)((copy(getproperty(v,fn)) for fn in propertynames(v))...)

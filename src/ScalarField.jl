#
# Scalar field structure
#
struct ScalarField{D} <: AbstractScalarField
    field::D
end

#generic setor (could add typing if concern with dimension compability) 
#ScalarField(dims::NTuple{N,Int64}) where {N} = ScalarField((zeros(dims...) for fn in fieldnames(ScalarField))...)
ScalarField(dims::NTuple{N,Int64}) where {N} = ScalarField((FieldData(zeros(dims...)) for fn in fieldnames(ScalarField))...)
ScalarField(n::Int64) = ScalarField((n,))
ScalarField(nx::Int64, ny::Int64) = ScalarField((nx, ny))
ScalarField(grid::AbstractGrid) = ScalarField(size(grid))
Adapt.@adapt_structure ScalarField

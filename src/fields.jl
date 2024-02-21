
struct GridData{T,B<:Backend} <: AbstractGridData{B}
    data::T
    backend::B
end
GridData(data) = GridData(current_backend(data), current_backend.value)
Base.size(g::GridData) = size(g.data)
Adapt.@adapt_structure GridData

struct FieldData{T,B<:Backend} <: AbstractFieldData{B}
    data::T
    backend::B
end

FieldData(data) = FieldData(current_backend(data), current_backend.value)
Adapt.@adapt_structure FieldData
# accesssor
(d::GridData)(i::Index, j::Index) = d.data[i, j]

(d::FieldData{T,B})(grid_data::AbstractGridDerivatives, i::Index, j::Index) where {T,B} = d.data[i, j]
(d::FieldData{T,B})(i::Index, j::Index) where {T,B} = d.data[i, j]
Base.ndims(::Type{FieldData{T,B}}) where {T,B} = ndims(T)
Base.size(f::FieldData) = size(f.data)
Base.copy(f::FieldData) = FieldData(copy(f.data))

#Base.zeros(f::FieldData) = zeros(f, current_backend)
Base.zeros(f::FieldData, backend::Backend) = FieldData(zeros(size(f.data)...))
# Base.ones(f::FieldData) = ones(f, current_backend)
Base.ones(f::FieldData) = FieldData(ones(size(f.data)...))

Base.copyto!(f::FieldData, args...) = copyto!(f.data, args...) 
CUDA.GPUArrays._copyto!(f::FieldData, args...) = CUDA.GPUArrays._copyto!(f.data, args...)
#generic setor (could add typing if concern with dimension compability) 
Base.setindex!(f::FieldData, args...) = setindex!(f.data, args...)
Base.getindex(v::FieldData, args...) = getindex(v.data, args...)


# Vector field structure
#
# we only consider 3 components because of curl 
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

#TensorField

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
# --------------------------------------------- #
prettytype(N::Type) = split(string(N), ".")[end]
# ----------------- display ------------------- #
function Base.show(io::IO, ::MIME"text/plain", f::Field)
   for fn in propertynames(f)
    println(io, " -- $fn --")
    println(io, getproperty(f,fn).data)
   end
end

# function Base.show(io::IO, f::Field)
    
#     print(io, "$(typeof(f).name.name)")
# end



# function Base.show(io::IO, ::MIME"text/plain", G::Type{T}) where T<:Field

#     print(io, "$(G.name.name)")
# end

# function Base.show(io::IO, G::Type{T}) where T<:Field
#     print(io, "$(G.name.name)")
# end

struct GridData{T} <: AbstractGridData{T}
    data::T
end

struct FieldData{T} <: AbstractFieldData{T}
    data::T
end

# accesssor
(d::GridData)(i, j) = d.data[i, j]

(d::FieldData{T})(grid_data::AbstractGridDerivatives, i, j) where {T} = d.data[i, j]
(d::FieldData{T})(i, j) where {T} = d.data[i, j]
Base.ndims(::Type{FieldData{T}}) where T = ndims(T)
Base.size(f::FieldData) = size(f.data)
Base.copyto!(f::FieldData, args...) = copyto!(f.data, args...) 
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

# --- display ---
function Base.show(io::IO, ::MIME"text/plain", f::Field)
   for fn in propertynames(f)
    println(io, " -- $fn --")
    println(io, getproperty(f,fn).data)
   end
end
prettytype(N::Type) = split(string(N), ".")[end]
function Base.show(io::IO, ::T) where {T<:Field}
    print(io, "$(pretty_type(T))")
end
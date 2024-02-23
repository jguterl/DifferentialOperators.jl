#
# Field data structure
#
struct FieldData{T,B<:Backend} <: AbstractFieldData{B}
    data::T
    backend::B
end

FieldData(data) = FieldData(current_backend(data), current_backend.value)
Adapt.@adapt_structure FieldData
# accesssor
(d::FieldData{T,B})(grid_data::AbstractCoordSpacings, i::Index, j::Index) where {T,B} = d.data[i, j]
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



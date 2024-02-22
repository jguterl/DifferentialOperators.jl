#
# Grid data structure definition -- why was this in fields????
#
struct GridData{T,B<:Backend} <: AbstractGridData{B}
    data::T
    backend::B
end
GridData(data) = GridData(current_backend(data), current_backend.value)
Base.size(g::GridData) = size(g.data)
Adapt.@adapt_structure GridData
(d::GridData)(i::Index, j::Index) = d.data[i, j]

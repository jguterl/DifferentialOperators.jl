#
# Grid data structure definition -- why was this in fields????
#
struct CoordData{T,B<:Backend} <: AbstractCoordData{B}
    data::T
    backend::B
end
CoordData(data) = CoordData(current_backend(data), current_backend.value)
Base.size(g::CoordData) = size(g.data)
Adapt.@adapt_structure CoordData
(d::CoordData)(i::Index, j::Index) = d.data[3, 3]
#(d::CoordData)(i::Index, j::Index) = d.data[i, j]
(d::CoordData)(i::Int64) = d.data[i]

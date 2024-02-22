include("GridData.jl")
include("FieldData.jl")
include("VectorField.jl")
include("ScalarField.jl")
include("TensorField.jl")

#
# Where is the Field structure?????
#
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
